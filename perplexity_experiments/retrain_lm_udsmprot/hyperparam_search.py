from sigopt import Connection
from train import generic_model, kwargs_defaults
import numpy as np
import argparse
import time
import os
import logging
import math
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)

transformations = {
        'batch_size': lambda x: x*32,
        'clip': lambda x: np.exp(x),
        'input_p': lambda x: np.exp(x),
        'output_p': lambda x: np.exp(x),
        'weight_p': lambda x: np.exp(x),
        'embed_p': lambda x: np.exp(x),
        'lr': lambda x: np.exp(x),
        'hidden_p': lambda x: np.exp(x),
        'wd': lambda x: np.exp(x),
        'dropout': lambda x: np.exp(x),
}

default_transfrom = lambda x: x
#needs the lambda because expects a factory, not a constant, so make a lambda that returns a constant
TRANSFORMATIONS_DICT = defaultdict(lambda : default_transfrom, transformations) 

kwargs_defaults['wandb_id'] = 'lm-enzyme-classification'
kwargs_defaults['pad_token'] = '<pad>'

def make_experiment(name: str = "EC classification LM"):
    conn = Connection(client_token="JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH")
    #define parameter space as list of tuples, and then encode to dict.
    space = [
            ('bs',          1,             8,            'int'),
            ('lr',          np.log(0.0001),np.log(20),   'double'),
            ('wd',          np.log(1e-40), np.log(0.001),'double'),
            ('dropout',     np.log(1e-40), np.log(0.6),  'double'),
            ('input_p',     np.log(1e-40), np.log(0.6),  'double'),
            ('output_p',    np.log(1e-40), np.log(0.6),  'double'),
            ('weight_p',    np.log(1e-40), np.log(0.6),  'double'),
            ('embed_p',     np.log(1e-40), np.log(0.6),  'double'),
            ('hidden_p',    np.log(1e-40), np.log(0.6),  'double'),            
            ('clip',        np.log(0.05),  np.log(1),    'double'),

            ]

    parameters =  [{'name': n, 'bounds' : {'min': mn, 'max':mx},'type': t} for n,mn,mx,t in space]

    experiment = conn.experiments().create(
        name=name,
        parameters = parameters,
        metrics = [dict(name ='Macro F1', objective ='maximize', strategy = 'optimize')],
        observation_budget=30,
        parallel_bandwidth=1,
        project="sigopt-examples"
        )
    print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)


#plasmodium
plasmodium_args = kwargs_defaults.copy()
plasmodium_args['from_scratch'] = False
plasmodium_args['pretrained_folder'] = '/zhome/1d/8/153438/experiments/results/best_models_25_08_2020/plasmodium/udsmprot'
plasmodium_args['epochs'] = 30 
plasmodium_args['metrics'] = ['accuracy','macro_f1'] 
plasmodium_args['lr'] = 0.001
plasmodium_args['lr_fixed'] = True 
plasmodium_args['bs'] = 32
plasmodium_args['lr_slice_exponent'] = 2.0
plasmodium_args['working_folder'] = '/work3/felteu/data/ec_prediction/balanced_splits/udsmprot/level_0/'
plasmodium_args['export_preds'] = True
plasmodium_args['eval_on_val_test'] = True
plasmodium_args['pad_token'] = "<pad>"
plasmodium_args['wandb_id'] =  'lm-enzyme-classification'
plasmodium_args['input_p'] = 0.65
plasmodium_args['output_p'] =  1.799743905199874e-22
plasmodium_args['weight_p'] = 0.013989742184768663
plasmodium_args['embed_p'] = 2.0115476819541153e-23
plasmodium_args['hidden_p'] = 4.537617849343064e-12
plasmodium_args['emb_sz'] = 192
plasmodium_args['nh'] = 384
plasmodium_args['nl'] = 2
plasmodium_args['pretrained_model_filename'] = "fastai_checkpoint_enc"
plasmodium_args['wd'] = 1.748385946935176e-30

#eukarya
eukarya_args = kwargs_defaults.copy()
eukarya_args['from_scratch'] = False
eukarya_args['pretrained_folder'] = '/zhome/1d/8/153438/experiments/results/best_models_25_08_2020/eukarya/udsmprot'
eukarya_args['epochs'] = 30 
eukarya_args['metrics'] = ['accuracy','macro_f1']
eukarya_args['lr_fixed'] = True 
eukarya_args['working_folder'] = '/work3/felteu/data/ec_prediction/balanced_splits/udsmprot/level_0/'
eukarya_args['export_preds'] = True
eukarya_args['eval_on_val_test'] = True
eukarya_args['lr_slice_exponent'] = 2.0
eukarya_args['pretrained_model_filename'] = "fastai_checkpoint_enc"

def evaluate_parameters(assignments: dict, org) -> float:
    '''Make and fit the model. Report validation perplexity.
    '''
    #build arg parser
    if org == 'plasmodium':
        args = plasmodium_args
    else:
        args = eukarya_args
    #transform assignments and add to namespace
    for parameter in assignments:
        parameter_mapped = TRANSFORMATIONS_DICT[parameter](assignments[parameter])
        args[parameter] = parameter_mapped

    result = generic_model(clas=True, **args)
    loss, accuracy, macro_f1 =result[0] #loss, accuracy, macro_f1

    # Obtain a metric for the dataset
    return [{'name': 'Macro F1', 'value' :macro_f1 }]


def test_parameter_space(experiment,num_runs: int, output_dir, org):
    '''get num_runs suggestions, train model with suggestion parameters, and report back results.
    Will also report suggestions as failed in case of failure'''
    for i in range(num_runs):


        suggestion = conn.experiments(experiment.id).suggestions().create()
        logger.info(f'Suggestion: {suggestion.id}')

        try:
            values = evaluate_parameters(suggestion.assignments, org)

            # Report an observation
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                values=values,
            )
            # Update the experiment object
            experiment = conn.experiments(experiment.id).fetch()
            logger.info(f'Reported: {suggestion.id}')
        except Exception as e:
            logger.info(e)
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                failed=True,
            )
            logger.info(f'Reported: {suggestion.id} as failed.')

        


if __name__ == '__main__':
    conn = Connection(client_token="JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH")
    hypparser = argparse.ArgumentParser(description='Run one Sigopt Suggestion')
    hypparser.add_argument('--output_dir', type=str, default ='/work3/felteu/hyperopt_runs_ec')
    hypparser.add_argument('--experiment_id', type = int, default = 326980)
    hypparser.add_argument('--organism', type=str, default = 'plasmodium')

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


    f_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

 

    #choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on: {device}')

    test_parameter_space(experiment, num_runs, output_dir, script_args.organism)

    # Fetch the best configuration and explore your experiment
    #all_best_assignments = conn.experiments(experiment.id).best_assignments().fetch()
    # Returns a list of dict-like Observation objects
    #best_assignments = all_best_assignments.data[0].assignments
    #print("Best Assignments: " + str(best_assignments))
    # Access assignment values as:
    #parameter_value = best_assignments['parameter_name']
    logger.info("Complete. Explore the experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")