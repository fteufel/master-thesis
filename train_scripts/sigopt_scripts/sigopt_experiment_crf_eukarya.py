#experiment for awd-lstm pretraining.

from sigopt import Connection
import sys
import numpy as np
sys.path.append("../..")
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from train_scripts.signal_peptide_prediction_crf_awd_lstm import main_training_loop
import argparse
import time
import os
import logging
import math
import torch
from train_scripts.sigopt_scripts.sigopt_utils import TRANSFORMATIONS_DICT

#patch the additional transformation needed here
TRANSFORMATIONS_DICT['classifier_hidden_size'] = lambda x: x*32

###Only run this here once
def make_experiment(name: str = "SP Prediction Eukarya AWD-LSTM"):
    conn = Connection(client_token="JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH")
    parameters=[
        dict(
        name="batch_size", bounds=dict( min=0, max=50 ),type="double")
        ],
    #define parameter space as list of tuples, and then encode to dict.
    space = [
            ('batch_size',             1,    8, 'int'),
            ('lr',                     np.log(0.0001),   np.log(20), 'double'),
            ('classifier_hidden_size',            1, 15, 'int'),
    ]

    parameters =  [{'name': n, 'bounds' : {'min': mn, 'max':mx},'type': t} for n,mn,mx,t in space]

    experiment = conn.experiments().create(
        name=name,
        parameters = parameters,
        metrics = [dict(name ='Global AUC', objective='maximize', strategy = 'optimize' ),
                   dict(name ='CS F1', objective ='maximize', strategy = 'optimize')],
        observation_budget=30,
        parallel_bandwidth=1,
        project="sigopt-examples"
        )
    print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)



def get_default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='AWD-LSTM language modeling')
    parser.add_argument('--data', type=str, default='/work3/felteu/data/signalp_5_data/famsa_225_partitions/',
                        help='location of the data corpus. Expects test, train and valid .txt')

    #args relating to training strategy.
    parser.add_argument('--lr', type=float, default=10,
                        help='initial learning rate')
    parser.add_argument('--lr_step', type = float, default = 0.9,
                        help = 'factor by which to multiply learning rate at each reduction step')
    parser.add_argument('--update_lr_steps', type = int, default = 6000,
                        help = 'After how many update steps to check for learning rate update')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--wait_epochs', type = int, default = 3,
                        help='Reduce learning rates after wait_epochs epochs without improvement')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--reset_hidden', type=bool, default=False,
                        help = 'Reset the hidden state after encounter of the tokenizer stop token')
    parser.add_argument('--log_interval', type=int, default=10000, metavar='N',
                        help='report interval')
    parser.add_argument('--output_dir', type=str,  default="/zhome/1d/8/153438/experiments/results/simple_sp_tagging/",
                        help='path to save logs and trained model')
    parser.add_argument('--wandb_sweep', type=bool, default=False,
                        help='wandb hyperparameter sweep: Override hyperparams with params from wandb')
    parser.add_argument('--resume', type=str,  default="/zhome/1d/8/153438/experiments//results/best_models_31072020/best_euk_model",
                        help='path of model to resume (directory containing .bin and config.json')
    parser.add_argument('--experiment_name', type=str,  default='CRF_sigopt',
                        help='experiment name for logging')
    parser.add_argument('--enforce_walltime', type=bool, default =True,
                        help='Report back current result before 24h wall time is over')


    #args for model architecture
    parser.add_argument('--classifier_hidden_size', type=int, default=128, metavar='N',
                        help='Hidden size of the classifier head MLP')
    parser.add_argument('--num_labels', type=int, default=4, metavar='N',
                        help='Number of labels for the classifier head')
    args = parser.parse_known_args()[0]
    return args

def evaluate_parameters(assignments: dict, static_options: dict) -> float:
    '''Make and fit the model. Report validation perplexity.
    '''
    #build arg parser
    args = get_default_args()
    #transform assignments and add to namespace
    for parameter in assignments:
        parameter_mapped = TRANSFORMATIONS_DICT[parameter](assignments[parameter])
        setattr(args,parameter,parameter_mapped )

    #add static options
    for key in static_options:
        setattr(args, key, static_options[key])

    best_loss, best_AUC, best_F1 = main_training_loop(args)

    # Obtain a metric for the dataset
    return [{'name': 'Global AUC', 'value' :best_AUC },{'name':'CS F1', 'value': best_F1}]


def test_parameter_space(experiment,num_runs: int, output_dir, data):
    for i in range(num_runs):

        #only necessary if it deviates from the default in the AWDLSTMconfig
        static_options = {'data': data,
                          'optimizer': 'sgd',
                          'reset_hidden': False,
                          'output_dir' : output_dir,
                          'wandb_sweep': False,
                          'experiment_name' : 'sigopt_parameter_run',
                        }

        suggestion = conn.experiments(experiment.id).suggestions().create()
        values = evaluate_parameters(suggestion.assignments, static_options)

        # Report an observation
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            values=values, #report perlexity
        )
        # Update the experiment object
        experiment = conn.experiments(experiment.id).fetch()
    


if __name__ == '__main__':
    conn = Connection(client_token="JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH")
    hypparser = argparse.ArgumentParser(description='Run one Sigopt Suggestion')
    hypparser.add_argument('--output_dir', type=str, default ='/work3/felteu/hyperopt_runs_crf')
    hypparser.add_argument('--experiment_id', type = int, default = 304075)
    hypparser.add_argument('--data', type = str, default = '/work3/felteu/data/signalp_5_data/famsa_225_partitions/')

    script_args = hypparser.parse_args()
    experiment = conn.experiments(script_args.experiment_id).fetch()
    num_runs = 1
    output_dir = script_args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #make unique output dir
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    output_dir = os.path.join(output_dir, 'sigopt_parameter_run'+time_stamp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    #choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on: {device}')

    test_parameter_space(experiment, num_runs, output_dir, script_args.data)

    # Fetch the best configuration and explore your experiment
    #all_best_assignments = conn.experiments(experiment.id).best_assignments().fetch()
    # Returns a list of dict-like Observation objects
    #best_assignments = all_best_assignments.data[0].assignments
    #print("Best Assignments: " + str(best_assignments))
    # Access assignment values as:
    #parameter_value = best_assignments['parameter_name']
    logger.info("Complete. Explore the experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")
