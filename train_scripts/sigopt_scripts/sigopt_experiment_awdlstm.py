#experiment for awd-lstm pretraining.

from sigopt import Connection
import sys
import numpy as np
sys.path.append("../..")
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from train_scripts.train_awdlstm_lm import main_training_loop
import argparse
import time
import os
import logging
import math
import torch
from train_scripts.sigopt_scripts.sigopt_utils import TRANSFORMATIONS_DICT




###Only run this here once
def make_experiment(name: str = "AWD-LSTM LM Eukarya"):
    conn = Connection(client_token="JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH")
    parameters=[
        dict(
        name="batch_size", bounds=dict( min=0, max=50 ),type="double")
        ],
    #define parameter space as list of tuples, and then encode to dict.
    space = [
            ('batch_size',             1,    8, 'int'),
            ('lr',                     np.log(1),   np.log(20), 'double'),
            ('lr_step',                np.log(0.05), np.log(1), 'double'),
            ('clip',                   np.log(0.05), np.log(1), 'double'),
            #('wait_epochs',            20, 400, 'int'), #lets always wait for 5
            ('bptt',                   1,    8, 'int'),
            ('wdecay',                 np.log(1e-40), np.log(0.001), 'double'),
            ('num_hidden_layers',      2,  5, 'int'),
            ('input_size',             1, 8, 'int'),
            ('hidden_size',            2, 19, 'int'),
            ('dropout_prob',           np.log(1e-40), np.log(0.6), 'double'),
            ('hidden_dropout_prob',    np.log(1e-40), np.log(0.6), 'double'),
            ('embedding_dropout_prob', np.log(1e-40), np.log(0.6), 'double'),
            ('weight_dropout_prob',    np.log(1e-40), np.log(0.6), 'double'),
            #('alpha',                  20, 400, 'double'),
            #('beta',                   20, 400, 'double')
    ]

    parameters =  [{'name': n, 'bounds' : {'min': mn, 'max':mx},'type': t} for n,mn,mx,t in space]

    experiment = conn.experiments().create(
        name=name,
        parameters = parameters,
        metrics = [dict(name ='Perplexity', objective='minimize', strategy = 'optimize' )],
        observation_budget=30,
        parallel_bandwidth=1,
        project="sigopt-examples"
        )
    print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)



def get_default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='AWD-LSTM language modeling')
    parser.add_argument('--data', type=str, default='../data/awdlstmtestdata/',
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
    parser.add_argument('--buffer_size', type= int, default = 5000000,
                        help = 'How much data to load into RAM (in bytes')
    parser.add_argument('--log_interval', type=int, default=10000, metavar='N',
                        help='report interval')
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')
    parser.add_argument('--wandb_sweep', type=bool, default=False,
                        help='wandb hyperparameter sweep: Override hyperparams with params from wandb')
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume (directory containing .bin and config.json')
    parser.add_argument('--experiment_name', type=str,  default='AWD_LSTM_LM',
                        help='experiment name for logging')
    parser.add_argument('--enforce_walltime', type=bool, default =True,
                        help='Report back current result before 24h wall time is over')
    #args for model architecture
    parser.add_argument('--num_hidden_layers', type=int, default=3, metavar='N',
                        help='report interval')
    parser.add_argument('--input_size', type=int, default=10, metavar='N',
                        help='Embedding layer size')
    parser.add_argument('--hidden_size', type=int, default=1000, metavar='N',
                        help='LSTM hidden size')
    parser.add_argument('--dropout_prob', type=float, default=0.4,
                        help='Dropout')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3,
                        help='Dropout between layers')
    parser.add_argument('--embedding_dropout_prob', type=float, default=0.1,
                        help='Dropout embedding layer')
    parser.add_argument('--input_dropout_prob', type=float, default=0.65,
                        help='Dropout input')
    parser.add_argument('--weight_dropout_prob', type=float, default=0.5,
                        help='Dropout LSTM weights')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Activation regularization beta')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Activation regularization alpha')
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

    best_loss = main_training_loop(args)

    # Obtain a metric for the dataset
    return best_loss


def test_parameter_space(experiment,num_runs: int, output_dir, data):
    for i in range(num_runs):

        #only necessary if it deviates from the default in the AWDLSTMconfig
        static_options = {'data': data,
                          'wait_epochs': 5,
                          'optimizer': 'sgd',
                          'reset_hidden': False,
                          'output_dir' : output_dir,
                          'wandb_sweep': False,
                          'resume' : False,
                          'experiment_name' : 'sigopt_parameter_run',
                        }

        suggestion = conn.experiments(experiment.id).suggestions().create()
        value = evaluate_parameters(suggestion.assignments, static_options)

        # Report an observation
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=math.exp(value), #report perlexity
        )
        # Update the experiment object
        experiment = conn.experiments(experiment.id).fetch()
    


if __name__ == '__main__':
    conn = Connection(client_token="JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH")
    hypparser = argparse.ArgumentParser(description='Run one Sigopt Suggestion')
    hypparser.add_argument('--output_dir', type=str, default ='/work3/felteu/hyperopt_runs')
    hypparser.add_argument('--experiment_id', type = int, default = 227405)
    hypparser.add_argument('--data', type = str, default = '/work3/felteu/data/splits/eukarya')

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
