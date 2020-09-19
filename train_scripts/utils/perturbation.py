# Copyright (c) Microsoft. All rights reserved.
# https://github.com/namisan/mt-dnn/blob/19beb901c97dac9fbc18ba451d18bd9790c72cc9/mt_dnn/perturbation.py
# removed some parameters that are not used.
# encoder_type, task_type (only do classification, no need for more complicated implementation. Can always go back to original file.)
# adapted model fwd pass wrapping to XLNetSequenceTaggingCRF
from copy import deepcopy
import torch
import logging
import random
from torch.nn import Parameter
from functools import wraps
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

#from .loss import stable_kl, SymKlCriterion

logger = logging.getLogger(__name__)

def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) *  epsilon
    noise.detach()
    noise.requires_grad_()
    return noise

def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    if reduce:
        return (p* (rp- ry) * 2).sum() / bs
    else:
        return (p* (rp- ry) * 2).sum()

class Criterion(_Loss):
    def __init__(self, alpha=1.0, name='criterion'):
        super().__init__()
        """Alpha is used to weight each loss term
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        return

class SymKlCriterion(Criterion):
    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction='batchmean') + \
            F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction='batchmean')
        loss = loss * self.alpha
        return loss

class SmartPerturbation():
    #Does not define new tensors, should work without nn.Module to push to GPU
    def __init__(self,
                 epsilon=1e-6,
                 step_size=1e-3,
                 noise_var=1e-5,
                 norm_p='inf',
                 k=1,
                 norm_level=0):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon 
        # eta
        self.step_size = step_size
        self.K = k
        # sigma
        self.noise_var = noise_var 
        self.norm_p = norm_p
        self.norm_level = norm_level > 0


    def _norm_grad(self, grad, sentence_level=False):
        if self.norm_p == 'l2':
            if sentence_level:
                direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon)
            else:
                direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon)
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction


#[self.mnetwork, logits] + inputs + [batch_meta.get('pairwise_size', 1)]
    def forward(self, model,
                logits,
                input_ids,
                attention_mask):
        '''Get SMART regularization term.
        Model needs to implement:
            - .transformer.word_embedding to access the embedding layer
            - accept inputs_embeds instead of inputs_ids in the fwd pass
            - have a flag return_logits in the fwd pass to return the prediction logits only
        '''
        # adv training
        #get embeddings only #model.transformer.embedding(input_ids)

        # init delta
        embed = model.transformer.word_embedding(input_ids)
        noise = generate_noise(embed, attention_mask, epsilon=self.noise_var)
        for step in range(0, self.K):
            adv_logits = model(inputs_embeds = embed+noise, return_logits = True)

            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False) 

            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            delta_grad = noise + delta_grad * self.step_size
            noise = self._norm_grad(delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()

        adv_logits = model(inputs_embeds = embed+noise, return_logits = True)

        #crossentropy
        adv_lc = SymKlCriterion()
        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        return adv_loss 