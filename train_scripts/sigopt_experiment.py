from sigopt import Connection
import sys
import numpy as np
sys.path.append("..")
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from train_scripts.train_awdlstm_lm import main_training_loop

#map quantized hyperparamters back to their real space
TRANSFORMATIONS_DICT = {
        'batch_size': lambda x: x*32,
        'bptt': lambda x: x*32,
        'clip': lambda x: np.exp(x),
        'dropout_prob': lambda x: np.exp(x),
        'embedding_dropout_prob': lambda x: np.exp(x),
        'hidden_dropout_prob': lambda x: np.exp(x),
        'weight_dropout_prob': lambda x: np.exp(x),
        'input_size': lambda x: x*64,
        'lr_step': lambda x: np.exp(x),
        'lr': lambda x: np.exp(x),
        'hidden_size': lambda x: x*64,
        'num_hidden_layers': lambda x: x,
        'wdecay': lambda x: np.exp(x),
}

###Only run this here once
def make_experiment():
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
        name="AWD-LSTM LM Eukarya",
        parameters = parameters,
        metrics = [dict(name ='Perplexity', objective='minimize', strategy = 'optimize' )],
        observation_budget=30,
        parallel_bandwidth=1,
        project="sigopt-examples"
        )
    print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)



def evaluate_parameters(assignments: dict, static_options: dict) -> float:
    '''Make and fit the model. Report validation perplexity.
    '''
    #build arg parser
    from argparse import Namespace

    args = Namespace()
    #transform assignments and add to namespace
    for parameter in assignments:
        setattr(args,parameter, TRANSFORMATIONS_DICT[parameter](assignments[parameter]))

    #add static options
    for key in static_options:
        setattr(args, key, static_options[key])

    best_loss = main_training_loop(args)

    # Obtain a metric for the dataset
    return best_loss


def test_parameter_space(experiment,num_runs: int):
    for i in range(num_runs):

        #only necessary if it deviates from the default in the AWDLSTMconfig
        static_options = {'data': '/work3/felteu/data/splits/eukarya',
                          'wait_epochs': 5,
                          'optimizer': 'SGD',
                          'reset_hidden': False,
                          'output_dir' : '/zhome/1d/8/153438/experiments/results/awdlstmhyperparamsearch',
                          'wandb_sweep': False,
                          'resume' : False,
                          'experiment_name' : 'sigopt_parameter_run',
                        }

        suggestion = conn.experiments(experiment.id).suggestions().create()
        value = evaluate_parameters(suggestion.assignments, static_options)


        # Report an observation
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        # Update the experiment object
        experiment = conn.experiments(experiment.id).fetch()
    


if __name__ == '__main__':
    conn = Connection(client_token="JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH")
    experiment = conn.experiments(227405).fetch()
    num_runs = 1

    test_parameter_space(experiment, num_runs)

    # Fetch the best configuration and explore your experiment
    all_best_assignments = conn.experiments(experiment.id).best_assignments().fetch()
    # Returns a list of dict-like Observation objects
    best_assignments = all_best_assignments.data[0].assignments
    print("Best Assignments: " + str(best_assignments))
    # Access assignment values as:
    #parameter_value = best_assignments['parameter_name']
    print("Explore your experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")
