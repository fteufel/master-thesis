'''
Distill ensemble model (6 models) into one.
This includes distilling the CRF.

The validation pass that is implemented is useless at the 
moment, as there is no validation set specified.

Targets (probs) can be produced by ensemble_prediction_bert.py using 
the --make_distillation_targets flag.
'''
import torch
import pickle
import wandb
import os
import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from models.multi_crf_bert import ProteinBertTokenizer, BertSequenceTaggingCRF
from transformers import BertConfig
import argparse
import numpy as np
import logging
import time
import h5py

class H5fDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, shuffle=False, return_weights=False, use_emissions_loss=False):
        self.h5f = h5py.File(file_path, 'r')
        self.input_ids = self.h5f['input_ids']
        self.input_masks = self.h5f['input_mask']
        self.pos_probs =  self.h5f['pos_probs'] if not use_emissions_loss else h5f['emissions']
        self.sp_classes= self.h5f['type']

        #always make the weights, so that the randomsampler can use them.
        values, counts = np.unique(self.sp_classes, return_counts=True)
        count_dict = dict(zip(values,counts))
        self.equal_freq_weights = np.array([1.0/count_dict[i] for i in self.sp_classes])  * len(self.sp_classes)

        if return_weights:
            self.weights = self.equal_freq_weights
        else:
            self.weights = np.ones_like(self.sp_classes)

        if shuffle:
            rand_idx = np.random.permutation(len(self.input_ids))
            self.input_ids=self.input_ids[()][rand_idx]
            self.input_masks = self.input_masks[()][rand_idx]
            self.pos_probs = self.pos_probs[()][rand_idx]
            self.sp_classes = self.sp_classes[()][rand_idx]
            self.weights = self.weights[rand_idx]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int):
        input_ids =  self.input_ids[index,:]
        input_mask = self.input_masks[index,:]
        pos_probs = self.pos_probs[index,:,:]
        classes = self.sp_classes[index]
        weight = self.weights[index]
        
        return input_ids, input_mask, pos_probs, classes, weight


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    return logger

def log_metrics(metrics_dict, split: str, step: int):
    '''Convenience function to add prefix to all metrics before logging.'''
    wandb.log({f"{split.capitalize()} {name.capitalize()}": value
                for name, value in metrics_dict.items()}, step=step)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model: torch.nn.Module, optimizer, train_loader, args, global_step):
    '''Iterate full dataloader to train.
    pos_probs are either the CRF marginals, or the emissions. Specify with args.use_emissions_loss.'''
    #process as batches
    all_losses = []
    all_losses_spi = []
    all_losses_spii = []
    all_losses_tat = []
    all_losses_tatlipo = []
    all_losses_spiii = []
    all_losses_other = []
    b_start = 0
    b_end = args.batch_size

    model.train()

    for i, b in enumerate(train_loader):
        ids, mask, true_pos_probs, classes, weight = b
        ids = ids.to(device)
        mask = mask.to(device)
        true_pos_probs = true_pos_probs.to(device)
        classes = classes.detach().numpy()
        weight = weight.to(device)

        optimizer.zero_grad()

        global_probs, pred_pos_probs, pos_preds, pred_emissions, mask = model(ids, input_mask=mask, return_emissions=True)

        #pos_probs: batch_size x seq_len x n_labels
        if args.use_emissions_loss:
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(pred_emissions/args.sm_temperature, dim=-1), 
                torch.nn.functional.softmax(true_pos_probs/args.sm_temperature, dim=-1),
                reduction='none')
            loss = loss.sum(dim=-1).mean(dim=-1)
        else:
            loss = torch.nn.functional.kl_div(torch.log(pred_pos_probs), true_pos_probs, reduction='none')
            loss = loss.sum(dim=-1).mean(dim=-1) #one loss per sample, sum over probability distribution axis
        #weight
        loss_weighted = loss * weight
        loss_weighted.mean().backward()
        if args.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        log_metrics({'KL Loss': loss.mean().item(), 'KL Loss weighted': loss_weighted.mean().item()},'train', global_step)
        global_step += 1


        loss = loss.detach().cpu().numpy()
        all_losses.append(loss)
        all_losses_spi.append(loss[classes == 1])#'SP'])
        all_losses_spii.append(loss[classes == 2])#'LIPO'])
        all_losses_tat.append(loss[classes == 3])#'TAT'])        
        all_losses_tatlipo.append(loss[classes == 4])#'TATLIPO'])
        all_losses_spiii.append(loss[classes == 5])#'PILIN'])
        all_losses_other.append(loss[classes == 0])#'NO_SP'])
        



    all_losses = np.concatenate(all_losses).mean()
    all_losses_spi = np.concatenate(all_losses_spi).mean()
    all_losses_spii = np.concatenate(all_losses_spii).mean()
    all_losses_tat = np.concatenate(all_losses_tat).mean()
    all_losses_tatlipo = np.concatenate(all_losses_tatlipo).mean()
    all_losses_spiii = np.concatenate(all_losses_spiii).mean()
    all_losses_other = np.concatenate(all_losses_other).mean()
    log_metrics({'KL All': all_losses,
                 'KL SP': all_losses_spi, 
                 'KL LIPO': all_losses_spii,
                 'KL TAT': all_losses_tat,
                 'KL TATLIPO': all_losses_tatlipo,
                 'KL PILIN': all_losses_spiii,
                 'KL Other': all_losses_other}, 'train', global_step)

    return sum([all_losses_other, all_losses_spi, all_losses_spii, all_losses_tat, all_losses_tatlipo, all_losses_spiii])/6, global_step

def validate(model: torch.nn.Module, optimizer, val_loader, args, global_step):
    '''Iterate full dataloader for validation metrics.'''
    #process as batches
    all_losses = []
    all_losses_spi = []
    all_losses_spii = []
    all_losses_tat = []
    all_losses_tatlipo = []
    all_losses_spiii = []
    all_losses_other = []
    b_start = 0
    b_end = args.batch_size

    for i, b in enumerate(val_loader):
        with torch.no_grad():

            ids, mask, true_pos_probs, classes, weight = b
            ids = ids.to(device)
            mask = mask.to(device)
            true_pos_probs = true_pos_probs.to(device)
            classes = classes.detach().numpy()
            weight = weight.to(device)

            global_probs, pred_pos_probs, pos_preds = model(ids, input_mask=mask)

            #pos_probs: batch_size x seq_len x n_labels
            loss = torch.nn.functional.kl_div(torch.log(pred_pos_probs), true_pos_probs, reduction='none')
            loss = loss.sum(dim=-1).mean(dim=-1) #one loss per sample, sum over probability distribution axis


            loss = loss.detach().cpu().numpy()
            all_losses.append(loss)
            all_losses_spi.append(loss[classes == 1])#'SP'])
            all_losses_spii.append(loss[classes == 2])#'LIPO'])
            all_losses_tat.append(loss[classes == 3])#'TAT'])        
            all_losses_tatlipo.append(loss[classes == 4])#'TATLIPO'])
            all_losses_spiii.append(loss[classes == 5])#'PILIN'])
            all_losses_other.append(loss[classes == 0])#'NO_SP'])
            

    all_losses = np.concatenate(all_losses).mean()
    all_losses_spi = np.concatenate(all_losses_spi).mean()
    all_losses_spii = np.concatenate(all_losses_spii).mean()
    all_losses_tat = np.concatenate(all_losses_tat).mean()
    all_losses_tatlipo = np.concatenate(all_losses_tatlipo).mean()
    all_losses_spiii = np.concatenate(all_losses_spiii).mean()
    all_losses_other = np.concatenate(all_losses_other).mean()
    log_metrics({'KL All': all_losses,
                 'KL SP': all_losses_spi, 
                 'KL LIPO': all_losses_spii,
                 'KL TAT': all_losses_tat,
                 'KL TATLIPO': all_losses_tatlipo,
                 'KL PILIN': all_losses_spiii,
                 'KL Other': all_losses_other}, 'val', global_step)

    return sum([all_losses_other, all_losses_spi, all_losses_spii, all_losses_tat, all_losses_tatlipo, all_losses_spiii])/6

def main_training_loop(args: argparse.ArgumentParser):


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger = setup_logger()
    f_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    f_handler.setFormatter(formatter)

    logger.addHandler(f_handler)

    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())


    #TODO get rid of this dirty fix once wandb works again
    global wandb
    import wandb
    if wandb.run is None and not args.crossval_run: #Only initialize when there is no run yet (when importing main_training_loop to other scripts)
        wandb.init(dir=args.output_dir, name=args.experiment_name)
    else:
        wandb=DecoyWandb()


    # Set seed
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        seed = args.random_seed
    else:
        seed = torch.seed()

    logger.info(f'torch seed: {seed}')
    wandb.config.update({'seed': seed})



    logger.info(f'Saving to {args.output_dir}')

    # Here we set up the config as always.
    # Instead of then loading a checkpoint, we use it to instantiate a full model.

    logger.info(f'Loading pretrained model in {args.resume}')
    config = BertConfig.from_pretrained(args.resume)

    if config.xla_device:
        setattr(config, 'xla_device', False)


    setattr(config, 'num_labels', args.num_seq_labels) 
    setattr(config, 'num_global_labels', args.num_global_labels)


    setattr(config, 'lm_output_dropout', args.lm_output_dropout)
    setattr(config, 'lm_output_position_dropout', args.lm_output_position_dropout)
    setattr(config, 'crf_scaling_factor', 1)
    setattr(config, 'use_large_crf', True) #legacy, parameter is used in evaluation scripts. Ensures choice of right CS states.


    if args.sp_region_labels:
        setattr(config, 'use_region_labels', True)




    #hardcoded for full model, 5 classes, 37 tags
    if args.constrain_crf and args.sp_region_labels:
        allowed_transitions = [
            
            #NO_SP
            (0,0), (0,1), (1,1), (1,2), (1,0), (2,1), (2,2), # I-I, I-M, M-M, M-O, M-I, O-M, O-O
            #SPI
            #3 N, 4 H, 5 C, 6 I, 7M, 8 O
            (3,3), (3,4), (4,4), (4,5), (5,5), (5,8), (8,8), (8,7), (7,7), (7,6), (6,6), (6,7), (7,8), 
            
            #SPII
            #9 N, 10 H, 11 CS, 12 C1, 13 I, 14 M, 15 O
            (9,9), (9,10), (10,10), (10,11), (11,11), (11,12), (12,15), (15,15), (15,14), (14,14), (14,13), (13,13), (13,14), (14,15),
            
            #TAT
            #16 N, 17 RR, 18 H, 19 C, 20 I, 21 M, 22 O
            (16,16), (16,17), (17,17), (17,16), (16,18), (18,18), (18,19), (19,19), (19,22), (22,22), (22,21), (21,21),(21,20), (20,20), (20,21), (21,22),
            
            #TATLIPO
            #23 N, 24 RR, 25 H, 26 CS, 27 C1, 28 I, 29 M, 30 O
            (23,23), (23,24), (24,24), (24,23), (23,25), (25,25), (25,26), (26,26), (26,27), (27,30), (30,30), (30,29), (29,29), (29,28), (28,28), (28,29),(29,30),
            
            #PILIN
            #31 P, 32 CS, 33 H, 34 I, 35 M, 36 O
            #TODO check transition from 33: to M or to O. Need to fix when making real h-region labels (so far ignoring TM info, just 10 pos h)
            (31,31), (31,32), (32,32), (32,33), (33,33), (33,36), (36,36), (36,35), (35,35), (35,34), (34,34), (34,35), (35,36)
            
        ]
        #            'NO_SP_I' : 0,
        #            'NO_SP_M' : 1,
        #            'NO_SP_O' : 2,
        allowed_starts = [0, 2, 3, 9, 16, 23, 31]
        allowed_ends = [0,1,2, 13,14,15, 20,21,22, 28,29,30, 34,35,36]

        setattr(config, 'allowed_crf_transitions', allowed_transitions)
        setattr(config, 'allowed_crf_starts', allowed_starts)
        setattr(config, 'allowed_crf_ends', allowed_ends)


    #setattr(config, 'gradient_checkpointing', True) #hardcoded when working with 256aa data
    if args.kingdom_as_token:
        setattr(config, 'kingdom_id_as_token', True) #model needs to know that token at pos 1 needs to be removed for CRF

    if args.global_label_as_input:
        setattr(config,'type_id_as_token', True)

    if args.remove_top_layers > 0:
        #num_hidden_layers if bert
        n_layers = config.num_hidden_layers if args.model_architecture == 'bert_prottrans' else config.n_layer
        if args.remove_top_layers > n_layers:
            logger.warning(f'Trying to remove more layers than there are: {n_layers}')
            args.remove_top_layers = n_layers

        setattr(config, 'num_hidden_layers' if args.model_architecture == 'bert_prottrans' else 'n_layer',n_layers-args.remove_top_layers)

    if args.resume_is_tagging_model:
        BertSequenceTaggingCRF.from_pretrained(args.resume)
        tokenizer = ProteinBertTokenizer.from_pretrained('resources/vocab_with_kingdom', do_lower_case=False)

    else:
        model = BertSequenceTaggingCRF.from_pretrained(args.resume, config=config)

        if args.kingdom_as_token:
            logger.info('Using kingdom IDs as word in sequence, extending embedding layer of pretrained model.')
            tokenizer = ProteinBertTokenizer.from_pretrained('resources/vocab_with_kingdom', do_lower_case=False)
            model.resize_token_embeddings(tokenizer.tokenizer.vocab_size)


    model.to(device)

    ## Setup data
    train_data =  H5fDataset(args.train_dataset,shuffle=args.shuffle_data, return_weights=args.weighted_loss, use_emissions_loss=args.use_emissions_loss)
    valid_data =  H5fDataset(args.valid_dataset,use_emissions_loss=args.use_emissions_loss)

    if args.use_weighted_random_sampling:
        sampler = torch.utils.data.WeightedRandomSampler(train_data.equal_freq_weights, len(train_data), replacement=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, sampler = sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size)

    #set up wandb logging, login and project id from commandline vars
    wandb.config.update(args)
    wandb.config.update(model.config.to_dict())

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    if os.path.isfile(os.path.join(args.resume,'optimizer_state.pt')):
        logger.info('Loading saved optimizer state')
        optimizer_state = torch.load(os.path.join(args.resume,'optimizer_state.pt'), map_location=device)
        optimizer.load_state_dict(optimizer_state) 




    logger.info('Model set up!')
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_parameters} trainable parameters')
    logger.info(f'Running model on {device}, not using nvidia apex')


    best_divergence = 1000000000000000
    global_step =0

    for epoch in range(1,args.epochs+1):


        epoch_kl_divergence, global_step = train(model, optimizer, train_loader, args, global_step)
        val_kl_divergence = validate(model, optimizer, valid_loader, args, global_step)
        
        if args.stop_on_validation:
            kl_div = val_kl_divergence
        else:
            kl_div = epoch_kl_divergence

        if kl_div<best_divergence:
            best_divergence=kl_div
            model.save_pretrained(args.output_dir)
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optimizer_state.pt'))
            logger.info(f'New best model with loss {kl_div}, training step {global_step}')






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train CRF on top of Pfam Bert')
    parser.add_argument('--train_dataset', type=str, default='sp_prediction_experiments/ensemble_probs_for_distillation.pkl',
                        help='location of the data hdf5 file')
    parser.add_argument('--valid_dataset', type=str, default='sp_prediction_experiments/ensemble_probs_for_distillation.pkl',
                        help='location of the validation hdf5 file')
    parser.add_argument('--shuffle_data', action='store_true', help='Randomly shuffle all data (does not work well with memory-mapped datasets)')
    parser.add_argument('--stop_on_validation', action='store_true', help='Use validation performance for early stopping.')
    parser.add_argument('--use_weighted_random_sampling', action='store_true', help='use a WeightedRandomSampler for the train set.')



            
    #args relating to training strategy.
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--optimizer', type=str,  default='adam',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_bert_distillation',
                        help='path to save logs and trained model')
    parser.add_argument('--resume', type=str,  default='Rostlab/prot_bert',
                        help='path of model to resume (directory containing .bin and config.json')
    
    # NOTE attempt to fix weird bug when resuming pretrained distillation checkpoints
    # When resuming from Rostlab/prot_bert, all good.
    # When resuming from previous distillation checkpoint, new checkpoint is 6x slower
    # altough config and file size are exactly the same still.
    parser.add_argument('--resume_is_tagging_model', action='store_true', help='checkpoint to be resumed is already a tagging model')

    parser.add_argument('--experiment_name', type=str,  default='Distill-Bert-CRF',
                        help='experiment name for logging')
    parser.add_argument('--crossval_run', action = 'store_true',
                        help = 'override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.')
    parser.add_argument('--log_all_final_metrics', action='store_true', help='log all final test/val metrics to w&b')


    parser.add_argument('--num_seq_labels', type=int, default=37)
    parser.add_argument('--num_global_labels', type=int, default=6)
    parser.add_argument('--global_label_as_input', action='store_true', help='Add the global label to the input sequence (only predict CS given a known label)')

    parser.add_argument('--weighted_loss', action='store_true', help='Balance loss for all classes')
    parser.add_argument('--use_emissions_loss', action='store_true', help='Use emissions instead of marginal probs. Need specify a pkl file as data that has "emissions"')
    parser.add_argument('--sm_temperature', type=float, default=1., help='Softmax temperature when training on emissions')

    parser.add_argument('--lm_output_dropout', type=float, default = 0.1,
                        help = 'dropout applied to LM output')
    parser.add_argument('--lm_output_position_dropout', type=float, default = 0.1,
                        help='dropout applied to LM output, drops full hidden states from sequence')
    parser.add_argument('--random_seed', type=int, default=None, help='random seed for torch.')

    #args for model architecture
    parser.add_argument('--model_architecture', type=str, default = 'bert_prottrans',
                        help ='which model architecture the checkpoint is for')
    parser.add_argument('--remove_top_layers', type=int, default=0, 
                        help='How many layers to remove from the top of the LM.')
    parser.add_argument('--kingdom_as_token', action='store_true', help='Kingdom ID is first token in the sequence')
    parser.add_argument('--sp_region_labels', action='store_true', help='Use labels for n,h,c regions of SPs.')
    parser.add_argument('--constrain_crf', action='store_true', help='Constrain the transitions of the region-tagging CRF.')


    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #make unique output dir in output dir
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    full_name = '_'.join([args.experiment_name, time_stamp])
    

    args.output_dir = os.path.join(args.output_dir, full_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    main_training_loop(args)
