

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import os
import sys
sys.path.append("..")
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet


from train_scripts.utils.signalp_dataset import RegionCRFDataset, pad_sequences
from train_scripts.downstream_tasks.metrics_utils import get_metrics_multistate, find_cs_tag
from train_scripts.utils.region_similarity import class_aware_cosine_similarities, get_region_lengths
from train_scripts.utils.cosine_similarity_regularization import compute_cosine_region_regularization
from models.multi_crf_t5 import ProteinT5Tokenizer, T5SequenceTaggingCRF


import numpy as np
import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

import transformers
from transformers import PreTrainedModel, Trainer, T5Config, HfArgumentParser, set_seed, TrainingArguments
from transformers.file_utils import is_torch_tpu_available
from transformers.integrations import is_fairscale_available
from transformers.models.fsmt.configuration_fsmt import FSMTConfig
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_utils import EvaluationStrategy, is_main_process
from transformers.trainer_pt_utils import get_tpu_sampler
from transformers.training_args import ParallelMode


if is_fairscale_available():
    from fairscale.optim import OSS

import json

logger = logging.getLogger(__name__)


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)

def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it
    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir
    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )



arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing (:obj:`float`, `optional`, defaults to 0):
            The label smoothing epsilon to apply (if not zero).
        sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to SortishSamler or not. It sorts the inputs according to lenghts in-order to minimizing the padding size.
        predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """

    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={"help": f"Which lr scheduler to use. Selected in {sorted(arg_to_scheduler.keys())}"},
    )





class TrainerDataset(RegionCRFDataset):
    '''
    Subclassed to replace collate_fn. This collate_fn returns dicts instead of lists.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        #unpack the list of tuples
        if  hasattr(self, 'sample_weights') and hasattr(self, 'kingdom_ids'):
            input_ids, label_ids,mask, global_label_ids, cleavage_sites, sample_weights, kingdom_ids = tuple(zip(*batch))
        elif hasattr(self, 'sample_weights'):
            input_ids, label_ids,mask, global_label_ids, sample_weights, cs = tuple(zip(*batch))
        elif hasattr(self, 'kingdom_ids'):
            input_ids, label_ids,mask, global_label_ids, kingdom_ids, cs = tuple(zip(*batch))
        else:
            input_ids, label_ids,mask, global_label_ids = tuple(zip(*batch))

        data = torch.from_numpy(pad_sequences(input_ids, 0))
        
        # ignore_index is -1
        targets = pad_sequences(label_ids, -1)
        targets = np.stack(targets)
        targets = torch.from_numpy(targets) 
        mask = torch.from_numpy(pad_sequences(mask, 0))
        global_targets = torch.tensor(global_label_ids)
        cleavage_sites = torch.tensor(cleavage_sites)

        return_dict = {'input_ids':data,
                        'targets_bitmap': targets,
                        'input_mask': mask,
                        'global_targets': global_targets,
                        'kingdom_ids': torch.tensor(kingdom_ids)

        }
                        
        return return_dict



class SPPredictionTrainer(Trainer):
    def __init__(self, config=None, data_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if config is None:
            assert isinstance(
                self.model, PreTrainedModel
            ), f"If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is {self.model.__class__}"
            self.config = self.model.config
        else:
            self.config = config

        self.data_args = data_args
        self.vocab_size = self.config.vocab_size

        if self.args.label_smoothing != 0 or (self.data_args is not None and self.data_args.ignore_pad_token_for_loss):
            assert (
                self.config.pad_token_id is not None
            ), "Make sure that `config.pad_token_id` is correcly defined when ignoring `pad_token` for loss calculation or doing label smoothing."

        if self.config.pad_token_id is None and self.config.eos_token_id is not None:
            logger.warn(
                f"The `config.pad_token_id` is `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for padding.."
            )

        self.loss_fn = None
        
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = self._get_lr_scheduler(num_training_steps)
        else:  # ignoring --lr_scheduler
            logger.warn("scheduler is passed to `SPPredictionTrainer`, `--lr_scheduler` arg is ignored.")

    def _get_lr_scheduler(self, num_training_steps):
        schedule_func = arg_to_scheduler[self.args.lr_scheduler]
        if self.args.lr_scheduler == "constant":
            scheduler = schedule_func(self.optimizer)
        elif self.args.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(self.optimizer, num_warmup_steps=self.args.warmup_steps)
        else:
            scheduler = schedule_func(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            if self.args.sortish_sampler:
                self.train_dataset.make_sortish_sampler(
                    self.args.per_device_train_batch_size,
                    distributed=(self.args.parallel_mode == ParallelMode.DISTRIBUTED),
                )

            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )


    def compute_loss(self, model,inputs):
        '''Compute region prediction loss.'''
        loss, global_probs, pos_probs, pos_preds = model(**inputs)                                            
        nh, hc = compute_cosine_region_regularization(pos_probs,
                                                      inputs['input_ids'][:,1:-1],
                                                      inputs['global_targets'],
                                                      inputs['input_mask'][:,1:-1])
        loss = loss+ nh.mean() * 0.5
        loss = loss+ hc.mean() * 0.5

        return loss



    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(**inputs) 


        loss = loss.mean().detach()
        if self.args.prediction_loss_only:
            return (loss, None, None)

        return (loss, global_probs, pos_preds)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor




def update_config_for_CRF_model(checkpoint):
    config = T5Config.from_pretrained(checkpoint)
    setattr(config, 'num_labels', 37) 
    setattr(config, 'num_global_labels', 6)

    setattr(config, 'lm_output_dropout', 0.1)
    setattr(config, 'lm_output_position_dropout', 0.05)
    setattr(config, 'crf_scaling_factor', 1)
    setattr(config, 'use_large_crf', True) #legacy, parameter is used in evaluation scripts. Ensures choice of right CS states.


    setattr(config, 'use_region_labels', True)


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

    setattr(config, 'kingdom_id_as_token', False) #model needs to know that token at pos 1 needs to be removed for CRF

    return config

def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))



def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    check_output_dir(training_args)


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    tokenizer = ProteinT5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False)
    config = update_config_for_CRF_model('Rostlab/prot_t5_xl_bfd')

    model = T5SequenceTaggingCRF.from_pretrained('Rostlab/prot_t5_xl_bfd', config=config)

    data = 'data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta'
    train_dataset = TrainerDataset(data, tokenizer= tokenizer, partition_id = [0],  
                                            add_special_tokens = True,return_kingdom_ids=True,
                                            make_cs_state = False,
                                            add_global_label = False,
                                            )
    eval_dataset = TrainerDataset(data, tokenizer = tokenizer, partition_id = [1], 
                                            add_special_tokens = True, return_kingdom_ids=True,
                                            make_cs_state = False,
                                            add_global_label = False)
    test_dataset = TrainerDataset(data, tokenizer = tokenizer, partition_id = [2], 
                                            add_special_tokens = True, return_kingdom_ids=True,
                                            make_cs_state = False,
                                            add_global_label = False)


    trainer = SPPredictionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn
        )

    all_metrics = {}


    logger.info('*** Train ***')
    print('Started training*******************************')
    train_result = trainer.train()
    metrics = train_result.metrics
    print('Ended training*******************************')

    metrics["train_n_objs"] = data_args.n_train

    trainer.save_model()  # this also saves the tokenizer

    if trainer.is_world_process_zero():
        handle_metrics("train", metrics, training_args.output_dir)
        all_metrics.update(metrics)

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == '__main__':
    main()