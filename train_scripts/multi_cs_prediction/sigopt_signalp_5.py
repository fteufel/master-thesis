import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") 

import argparse
import shutil
import logging
from train_scripts.multi_cs_prediction.train_eukarya_signalp_5 import main_training_loop
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





transformations = {
        'hidden_size': lambda x: x*32,
        'n_filters': lambda x: x*32,

}
default_transform = lambda x: x
#needs the lambda because expects a factory, not a constant, so make a lambda that returns a constant
TRANSFORMATIONS_DICT = defaultdict(lambda : default_transform, transformations) 

def make_argparse_object(parameter_dict: dict, output_dir: str):

    parser = argparse.ArgumentParser(description='Train CRF on top of Bert')


    #statics
    parser.add_argument('--data', type=str, default='/work3/felteu/multi_cs_data/SPs_dataset_first_CS.fasta')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--optimizer', type=str,  default='adam',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--experiment_name', type=str,  default='SignalP5_crossvalidation_run')
    parser.add_argument('--crossval_run', default=True,
                        help = 'override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.')
    parser.add_argument('--log_all_final_metrics', action='store_true', help='log all final test/val metrics to w&b')
    parser.add_argument('--sp_region_labels', action='store_true', help = 'Use region labels instead of standard signalp labels')
    parser.add_argument('--use_signalp6_labels', action='store_true')
    parser.add_argument('--num_seq_labels', type=int, default=6)
    parser.add_argument('--num_global_labels', type=int, default=2)

    #run-dependent
    parser.add_argument('--test_partition', type = int, default = 0)
    parser.add_argument('--validation_partition', type = int, default = 1)
    parser.add_argument('--output_dir', type=str,  default=output_dir)

    #not part of original hyperparameter space
    parser.add_argument('--dropout_conv1', type=float, default=0.15)
    parser.add_argument('--dropout_input', type=float, default=0.25)
    #hyperparameters
    #(learning rate, LSTM hidden units, number of convolutional filters, convolutional filter width)

    #TODO could make those two part of search space, by seems they are not in the list
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip', type=float, default=parameter_dict['clip'])
    
    parser.add_argument('--lr', type=float, default=parameter_dict['lr'])
    parser.add_argument('--hidden_size', type=float, default=parameter_dict['hidden_size'])
    parser.add_argument('--n_filters',type=float, default=parameter_dict['n_filters'])
    parser.add_argument('--filter_size',type=float, default=parameter_dict['filter_size'])

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
        save_dir = os.path.join(base_dir, f'sp5_test_{test_partition}_val_{validation_partition}')
        setattr(args, 'output_dir', save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #call main
        setattr(args, 'experiment_name', f'crossval_{test_partition}_{validation_partition}_{time.strftime("%Y-%m-%d-%H-%M",time.gmtime())}')
        best_mcc_det, best_mcc_cs = main_training_loop(args)
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

    space = [
            ('lr',                   1e-4,            0.05,   'double', 'log'),
            ('hidden_size',          1,               16,      'int',  None),
            ('n_filters',            1,               16, 'int',       None),
            ('filter_size',          2,               5, 'int',       None),
            ('clip',                 0.05,             1, 'double',   'log'),
    ]

#{"name": "model-name","type": "categorical","categorical_values": ["resnet18", "resnet152"]},
#{"name": "lr", "type": "double", "bounds": { "min" : 0.001, "max": 0.01 }}
#]

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
        observation_budget=20,
        parallel_bandwidth=5,
        project="multi_cs_sp5"
        )
    print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)

