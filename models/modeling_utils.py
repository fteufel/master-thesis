# Felix July 2020
# Modular head models to use LMs on tasks.

import torch
import torch.nn as nn
import typing
from typing import List, Tuple

#TODO get mask from outside


class CRFSequenceTaggingHead(nn.Module):
    ''' CRF Module.
    Based on Bi-LSTM CRF, but makes no assumptions about the model providing the features. Can be anything, 
    as long as it is of shape [sequence_length, batch_size, feature_dim].

    Inputs:
        constraints: List of (int,int) tuples, defining which transitions are impossible
        input_dim: dimension of the input features
        num_tags: number of tags to be assigned
    '''
    def __init__(self, input_dim: int, num_tags: int, constraints: List[Tuple[int, int]] = None):
        super().__init__()
        
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transition_matrix = torch.nn.Parameter(torch.empty(num_tags, num_tags))
        self.end_probs = nn.Parameter(torch.empty(num_tags))#probability for each tag at the end of the sequence
        self.start_probs = nn.Parameter(torch.empty(num_tags)) #probability for each tag at the start of the sequence
        self.num_tags = num_tags
        self.features_to_tag = nn.Linear(input_dim, num_tags)

        if constraints is not None:
            for i, j in constraints:
                self.transition_matrix.data[i,j] = -10000

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None, targets: torch.Tensor = None):
        ''' Performs forward-backwards algorithm to compute marginal probabilities at each position.
        Performs Viterbi algorithm to get most likely sequence.
        Computations performed in log space.
        Inputs: 
            features: `(seq_len, batch_size, input_dim)` feature tensor
            target_labels: `(seq_len, batch_size)` tensor of true tags.
            true_lengths : `(batch_size,)` tensor of true sequence lengths without padding, to mask the CRF decoding

        Returns:
            (loss) : Crossentropy between marginal probabilities and true tags
            probs : Marginal probabilities at each position (forward-backward)
            tags: Tagged sequence (Viterbi algorithm decoding)

        '''
        if mask is None:
            mask = torch.ones(features.shape[0], features.shape[1], dtype = torch.uint8)
            
        #map input features to tag dimension
        emissions = self.features_to_tag(features)

        #NOTE get mask from dataloader collate_fn, no need to recreate here. maybe need sometime
        #mask = torch.ones(emissions.shape[0],emissions.shape[1]).byte()
        #if true_lengths is not None:
            # mask: (seq_length, batch_size). ignore = 0, compute = 1
        #    range_tensor = mask.cumsum(dim = 0)
        #    bool_mask = true_lengths.unsqueeze(0) >= range_tensor
        #    mask = bool_mask.type(torch.uint8)

        #get forward alpha
        alpha = self._compute_log_alpha(emissions, mask, run_backwards=False)
        #get backward beta
        beta = self._compute_log_alpha(emissions, mask, run_backwards=True)
        #combine
        z = torch.logsumexp(alpha[alpha.size(0)-1] + self.end_probs, dim=1)
        probs = alpha + beta - z.view(1, -1, 1)
        probs = torch.exp(probs)

        #viterbi decode
        tags = self._viterbi_decode(emissions, mask)

        outputs = (probs,tags)

        #get cross entropy loss at each position
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index= -1)
            loss = loss_fct(probs.view(-1, self.num_tags), targets.view(-1)) #-1 to flatten seq_len and batch_size
            outputs = loss + outputs

        return outputs

    def _compute_log_alpha(self,
                            emissions: torch.FloatTensor,
                            mask: torch.ByteTensor,
                            run_backwards: bool) -> torch.FloatTensor:
        '''https://github.com/kmkurn/pytorch-crf/blob/ac68deaf6d28c6a646ae455e7e3f55c29bfff5f3/torchcrf/__init__.py
        Function to compute alpha or beta with forwards-backwards algorithm
        '''
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.size()[:2] == mask.size()
        assert emissions.size(2) == self.num_tags
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()
        broadcast_transitions = self.transition_matrix.unsqueeze(0)  # (1, num_tags, num_tags)
        emissions_broadcast = emissions.unsqueeze(2)
        seq_iterator = range(1, seq_length)

        if run_backwards:
            # running backwards, so transpose
            broadcast_transitions = broadcast_transitions.transpose(1, 2) # (1, num_tags, num_tags)
            emissions_broadcast = emissions_broadcast.transpose(2,3)

            # the starting probability is end_probs if running backwards
            log_prob = [self.end_probs.expand(emissions.size(1), -1)]

            # iterate over the sequence backwards
            seq_iterator = reversed(seq_iterator)
        else:
            # Start transition score and first emission
            log_prob = [emissions[0] + self.start_probs.view(1, -1)]


        for i in seq_iterator:
            # Broadcast log_prob over all possible next tags
            broadcast_log_prob = log_prob[-1].unsqueeze(2)  # (batch_size, num_tags, 1)
            # Sum current log probability, transition, and emission scores
            score = broadcast_log_prob + broadcast_transitions + emissions_broadcast[i]  # (batch_size, num_tags, num_tags)
            # Sum over all possible current tags, but we're in log prob space, so a sum
            # becomes a log-sum-exp
            score = torch.logsumexp(score, dim=1) #(batch_size, num_tags)
            # Set log_prob to the score if this timestep is valid (mask == 1), otherwise
            # copy the prior value
            log_prob.append(score * mask[i].unsqueeze(1) +
                            log_prob[-1] * (1.-mask[i]).unsqueeze(1))

        if run_backwards:
            log_prob.reverse()

        return torch.stack(log_prob)


    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_probs + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transition_matrix + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_probs

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list