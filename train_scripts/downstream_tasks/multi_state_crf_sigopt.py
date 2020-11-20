'''
Jose 19/09/2020: Best start one batch job per hyperparameter set.
Choose which outer fold, then train 4 inner fold models and report average.

Gets 1 suggestion from SigOpt, trains 4 models, and reports average.
If the first model has a metric of 0, the other 3 models are skipped. It is assumed
that these hyperparameters are not trainable for any partitions (lr too high or whatever).

w&b logging is disabled because there's a bug when trying to log multiple runs from one script. reported.

Hyperparameters to optimize:
 - Dropout
 - Position-wise dropout
 - learning rate
 - batch size
 - kingdom embedding size
 - positive sample weight
'''
import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") 

import argparse
import shutil
import logging
from train_scripts.downstream_tasks.train_multistate_crf_model import main_training_loop
from sigopt import Connection
import numpy as np
from collections import defaultdict
import os
import wandb
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)


bool_dict = {'true':True, 'false':False}
transformations = {
        'batch_size': lambda x: x*20,
        'use_weighted_kingdom_sampling': lambda x: bool_dict[x],

}
default_transform = lambda x: x
#needs the lambda because expects a factory, not a constant, so make a lambda that returns a constant
TRANSFORMATIONS_DICT = defaultdict(lambda : default_transform, transformations) 

def make_argparse_object(parameter_dict: dict, output_dir: str):

    parser = argparse.ArgumentParser(description='Train CRF on top of Bert')

    #hyperparameters
    parser.add_argument('--lr', type=float, default=parameter_dict['lr'])
    parser.add_argument('--batch_size', type=int, default=parameter_dict['batch_size'])
    parser.add_argument('--lm_output_dropout', type=float, default = parameter_dict['lm_output_dropout'])
    parser.add_argument('--lm_output_position_dropout', type=float, default = parameter_dict['lm_output_position_dropout'])
    parser.add_argument('--crf_scaling_factor', type=int, default=parameter_dict['crf_scaling_factor'])
    parser.add_argument('--use_weighted_kingdom_sampling', type=bool, default=parameter_dict['use_weighted_kingdom_sampling'])

    #run-dependent
    parser.add_argument('--test_partition', type = int, default = 0)
    parser.add_argument('--validation_partition', type = int, default = 1)
    parser.add_argument('--output_dir', type=str,  default=output_dir)

    #statics
    parser.add_argument('--data', type=str, default='/zhome/1d/8/153438/experiments/master-thesis/data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta',)
    parser.add_argument('--sample_weights', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--wdecay', type=float, default=1.2e-6)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--optimizer', type=str,  default='smart_adamax')
    parser.add_argument('--resume', type=str,  default='Rostlab/prot_bert')
    parser.add_argument('--experiment_name', type=str,  default='crossvalidation_run')
    parser.add_argument('--crossval_run', type=bool, default=True)
    parser.add_argument('--log_all_final_metrics', action='store_true')
    parser.add_argument('--eukarya_only', type=bool, default=False)
    parser.add_argument('--archaea_only', type=bool, default=False)
    parser.add_argument('--num_seq_labels', type=int, default=37)
    parser.add_argument('--num_global_labels', type=int, default=6)
    parser.add_argument('--global_label_as_input', action='store_true')
    parser.add_argument('--region_regularization_alpha', type=float, default=0.5)
    parser.add_argument('--use_sample_weights', action = 'store_true')
    parser.add_argument('--use_random_weighted_sampling', type=bool, default=False) 
    parser.add_argument('--positive_samples_weight', type=float, default =None)
    parser.add_argument('--average_per_kingdom', default=True)
    parser.add_argument('--model_architecture', type=str, default = 'bert_prottrans')
    parser.add_argument('--use_rnn', action='store_true')
    parser.add_argument('--classifier_hidden_size', type=int, default=0, metavar='N')
    parser.add_argument('--remove_top_layers', type=int, default=1) 
    parser.add_argument('--kingdom_embed_size', type=int, default=0)
    parser.add_argument('--use_cs_tag', action='store_true')

    parser.add_argument('--kingdom_as_token', default=True)
    parser.add_argument('--sp_region_labels', default=True)
    parser.add_argument('--constrain_crf', default=True)

    parser.add_argument('--random_seed', type=int, default=np.random.randint(99999999))


    args_out = parser.parse_known_args()[0]

    full_name ='_'.join([args_out.experiment_name, 'test', str(args_out.test_partition), 'valid', str(args_out.validation_partition), 
                        time.strftime("%Y-%m-%d-%H-%M",time.gmtime())])
    args_out.output_dir = os.path.join(args_out.output_dir, full_name)
    if not os.path.exists(args_out.output_dir):
        os.makedirs(args_out.output_dir)
    return args_out


def cross_validate(test_partition, cli_args):
    conn = Connection(client_token="JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH")
    experiment = conn.experiments(cli_args.experiment_id).fetch()
    #build parameter space
    all_ids = [0,1,2,3,4]
    print(f'testing on {test_partition}, leave out.')
    all_ids.remove(test_partition)

    #get parameters from sigopt
    suggestion = conn.experiments(experiment.id).suggestions().create()
    assignments_transformed = {}
    for parameter in suggestion.assignments:
        assignments_transformed[parameter] = TRANSFORMATIONS_DICT[parameter](suggestion.assignments[parameter])

    logger.info(assignments_transformed)
    
    args = make_argparse_object(assignments_transformed, os.path.join(cli_args.output_dir))
    base_dir = os.path.join(cli_args.output_dir, f'crossval_run_{test_partition}_sigopt_{suggestion.id}')
    if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    f_handler = logging.FileHandler(os.path.join(base_dir, 'log.txt'))
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    logger.info(f'Starting validation loop. Training {len(all_ids)} models.')
    logger.info(vars(args))
    result_list_det = []
    result_list_cs  = []
    for validation_partition in all_ids:
        #run = wandb.init(reinit=True, name=f'Test{test_partition}_Val{validation_partition}_{suggestion.id}')
        logger.info('Initialized new w&b run.')
        setattr(args, 'validation_partition', validation_partition)
        setattr(args, 'test_partition', test_partition)

        #set output directory for this run- save all, so that I don't have to retrain once i finish the search
        save_dir = os.path.join(base_dir, f'test_{test_partition}_val_{validation_partition}')
        setattr(args, 'output_dir', save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #call main
        setattr(args, 'experiment_name', f'crossval_{test_partition}_{validation_partition}_{time.strftime("%Y-%m-%d-%H-%M",time.gmtime())}')

        # call main until a run with a good seed is found and completed
        found_good_seed = False
        while found_good_seed == False:
            best_mcc_det, best_mcc_cs, run_completed = main_training_loop(args)
            found_good_seed = run_completed
            logger.info(f'Seed {args.random_seed} good for region finding: {found_good_seed}')

        result_list_det.append(best_mcc_det)
        result_list_cs.append(best_mcc_cs)
        logger.info(f'partition {validation_partition}: MCC Detection {best_mcc_det}, MCC CS {best_mcc_cs}')
        #wandb.config.update({'sigopt_run': suggestion.id})

        #run.finish()
        logger.info(f'Finished partition {validation_partition}.')
        if best_mcc_det == 0 or best_mcc_cs ==0:
            logger.info('Model was not trainable for these hyperparameters. Do not train other inner folds, report performance of 0 and end.')
            break
    #take mean over all partitions
    average_det = sum(result_list_det)/len(result_list_det)
    average_cs = sum(result_list_cs)/len(result_list_cs)
    std_det = np.array(result_list_det).std()
    std_cs  = np.array(result_list_cs).std()


    values = [{'name': 'MCC Detection', 'value' :average_det, 'value_stddev':std_det }, 
              {'name': 'MCC CS', 'value' :average_cs, 'value_stddev': std_cs}]
    conn.experiments(experiment.id).observations().create(
                                                suggestion=suggestion.id,
                                                values=values,
                                                )
    logger.info(values)
    logger.info(f'Reported: {suggestion.id}')

    

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-validate the model.')
    parser.add_argument('--output_dir', type=str,
                        help='path to save logs and trained model')
    parser.add_argument('--test_partition', type=int, default=0,
                        help='Test partition of the outer loop. Provide as argument, so can run 5 jobs in parallel.')
    parser.add_argument('--experiment_id', type =int, default = 327937, help = 'SigOpt ID of experiment for this test_partition')

    args = parser.parse_args()
    cross_validate(args.test_partition, args)



def make_experiment(name: str = "SP Prediction Crossvalidation"):
    conn = Connection(client_token="JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH")

    #define parameter space as list of tuples, and then encode to dict.
    space = [
            ('batch_size',                  1,               6,      'int',    None),
            ('lr',                          1e-6,            1e-4,   'double', 'log'),
            ('lm_output_dropout',           0.0001,          0.6,    'double', 'log'),
            ('lm_output_position_dropout',  0.0001,          0.6,    'double', 'log'),
            ('crf_scaling_factor',          0.01,            1,      'double',  None),
            ('use_weighted_kingdom_sampling','true',         'false',    'categorical', None)
    ]


    parameters = []
    for n,mn,mx,t,transform in space:
        if t == 'categorical':
            param = {'name':n, 'categorical_values': [mn, mx], 'type': t}
        else:
            param = {'name': n, 'bounds' : {'min': mn, 'max':mx},  'type': t}

        if transform is not None:
            param['transformation'] = transform
        parameters.append(param)

    #parameters =  [{'name': n, 'bounds' : {'min': mn, 'max':mx},'type': t, 'transformation':transform} for n,mn,mx,t, transform in space]

    experiment = conn.experiments().create(
        name=name,
        parameters = parameters,
        metrics = [dict(name ='MCC Detection', objective='maximize', strategy = 'optimize' ),
                   dict(name ='MCC CS', objective='maximize', strategy = 'optimize' )
                    ],
        observation_budget=25,
        parallel_bandwidth=5,
        project="signalp_6_models"
        )
    print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)

