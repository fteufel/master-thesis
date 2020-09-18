'''
Hyperparameters to optimize:
 - Dropout
 - Position-wise dropout
 - learning rate
 - batch size
 - classifier size
'''
import argparse
import shutil
import logging
from train_large_crf_model import main_training_loop
from sklearn.model_selection import ParameterSampler
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)

PARAMETER_SPACE = { 'lm_output_dropout': loguniform(0.0001, 0.6),
                    'lm_output_position_dropout': loguniform(0.0001, 0.6),
                    'lr': uniform(0.00001, 0.0002),
                    'classifier_hidden_size': [128, 256, 512],
                    'batch_size': [20, 40, 60, 80, 100],
                    'clip': loguniform(0.05, 1),
                    'wdecay': loguniform(1e-40, 1e-4)
                    }

def make_argparse_object(parameter_dict, output_dir):
    parser = argparse.ArgumentParser(description='Train CRF on top of Pfam Bert')
    #statics
    parser.add_argument('--data', type=str, default='/work3/felteu/data/signalp_5_data/full/train_set.fasta')
    parser.add_argument('--model_architecture', type=str, default = 'xlnet')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--optimizer', type=str,  default='adamax')
    parser.add_argument('--resume', type=str,  default='Rostlab/prot_xlnet')
    parser.add_argument('--experiment_name', type=str,  default='crossvalidation_run')
    parser.add_argument('--enforce_walltime', type=bool, default =True)

    #run-dependent
    parser.add_argument('--test_partition', type = int, default = 0)
    parser.add_argument('--validation_partition', type = int, default = 1)
    parser.add_argument('--output_dir', type=str,  default=output_dir)
    #hyperparameters
    parser.add_argument('--lr', type=float, default=parameter_dict['lr'])
    parser.add_argument('--clip', type=float, default=parameter_dict['clip'])
    parser.add_argument('--batch_size', type=int, default=parameter_dict['batch_size'])
    parser.add_argument('--wdecay', type=float, default=parameter_dict['wdecay'])
    parser.add_argument('--lm_output_dropout', type=float, default = parameter_dict['lm_output_dropout'])
    parser.add_argument('--lm_output_position_dropout', type=float, default = parameter_dict['lm_output_position_dropout'])
    parser.add_argument('--classifier_hidden_size', type=int, default=parameter_dict['classifier_hidden_size'])

    args_out = parser.parse_known_args()[0]

    full_name ='_'.join([args_out.experiment_name, 'test', str(args_out.test_partition), 'valid', str(args_out.validation_partition)])
    args_out.output_dir = os.path.join(args_out.output_dir, full_name)
    if not os.path.exists(args_out.output_dir):
        os.makedirs(args_out.output_dir)
    return args_out


def cross_validate(test_partition, cli_args):
    #build parameter space
    sampler = ParameterSampler(PARAMETER_SPACE, n_iter=5)
    all_ids = [0,1,2,3,4]
    print(f'testing on {test_partition}, leave out.')
    all_ids.remove(test_partition)
    parameters_to_test = list(sampler) #list of 5 different hyperparameter sets (dicts)

    #for each parameter set, get 5-fold crossvalidated performance
    # 5*5 = 25 => this wont fit in a single script.
    best_performance = 0
    for i, parameter_set in enumerate(parameters_to_test):
        logger.info(f'Cross-validate parameter set {i}')

        save_name = f'hyperparam_set_{i}'
        args = make_argparse_object(parameter_set, os.path.join(cli_args.output_dir, save_name))
        result_list = []

        for validation_partition in all_ids:
            setattr(args, 'validation_partition', validation_partition)
            setattr(args, 'test_partition', test_partition)
            #call main
            setattr(args, 'experiment_name', f'crossval_{test_partition}_{validation_partition}_param_set_{i}')
            best_validation_metric = main_training_loop(args)

        #take mean over all partitions
        average_performance = sum(result_list)/len(result_list)
        parameters_to_test[i]['Performance'] = average_performance
        logger.info(f'set {i}: mcc sum {average_performance}')
        #rename save of current models or delete - need to save memory.
        if average_performance > best_performance:
            shutil.move(os.path.join(args.output_dir,save_name), os.path.join(args.output_dir, save_name))
        else:
            #if not better, get rid of the checkpoints right away.
            shutil.rmtree(os.path.join(args.output_dir,save_name))
    

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-validate the model.')
    parser.add_argument('--output_dir', type=str,
                        help='path to save logs and trained model')
    parser.add_argument('--test_partition', type=int, default=0,
                        help='Test partition of the outer loop. Provide as argument, so can run 5 jobs in parallel.')

    args = parser.parse_args()

    cross_validate(args.test_partition, args)