from collections import defaultdict
import numpy as np
from typing import Callable
from argparse import Namespace

CONNECTION_TOKEN = "JNKRVPXKVSKBZRRYPWKZPGGZZTXECFUOLKMKHYYBEXTVXVGH"

#map quantized/ log trainsformed hyperparameters back to their real space
transformations = {
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
        'hyperbolic': lambda x: x,
}
default_transfrom = lambda x: x

#needs the lambda because expects a factory, not a constant, so make a lambda that returns a constant
TRANSFORMATIONS_DICT = defaultdict(lambda : default_transfrom, transformations) 


def evaluate_parameters(assignments: dict, static_options: dict, training_loop: Callable, args: Namespace ) -> float:
    '''Make and fit the model. Report validation perplexity.
    Puts everyting into namespace `args` and calls `training loop` with `args` as only argument 
    '''
    #transform assignments and add to namespace
    for parameter in assignments:
        parameter_mapped = TRANSFORMATIONS_DICT[parameter](assignments[parameter])
        setattr(args,parameter,parameter_mapped )

    #add static options
    for key in static_options:
        setattr(args, key, static_options[key])

    best_loss = training_loop(args)

    # Obtain a metric for the dataset
    return best_loss