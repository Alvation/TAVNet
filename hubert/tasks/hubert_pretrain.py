# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from argparse import Namespace
from dataclasses import dataclass, field
from fairseq import search
from fairseq.data import Dictionary,encoders
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

from hubert.datasets.hubert_dataset import HubertDataset
from .sequence_generator import SequenceGenerator
logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )

class LabelEncoderS2SToken(object):
    def __init__(self, dictionary: Dictionary, bpe_tokenizer) -> None:
        self.bpe_tokenizer = bpe_tokenizer
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        label = self.bpe_tokenizer.encode(label.lower())
        return self.dictionary.encode_line(
            label, append_eos=True, add_if_not_exist=False,
        ).long()

    def decode(self, tok, symbols_ignore=None):
        tok = self.dictionary.string(tok, extra_symbols_to_ignore=symbols_ignore)
        if self.bpe_tokenizer:
            tok = self.bpe_tokenizer.decode(tok)
        return tok

@dataclass
class HubertPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys " "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    is_s2s: bool=field(default=False, metadata={'help': 'seq2seq fine-tuning only'})
    tokenizer_bpe_name: Optional[str] = field(default=None, metadata={'help': 'tokenizer model name'})
    tokenizer_bpe_model: Optional[str] = field(default=None, metadata={'help': 'tokenizer model path'})

@register_task("hubert_seq2seq_pretraining", dataclass=HubertPretrainingConfig)
class HubertS2SPretrainingTask(FairseqTask):

    cfg: HubertPretrainingConfig

    def __init__(
        self,
        cfg: HubertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"HubertPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
            if cfg.is_s2s:
                self.state.add_factory("s2s_tokenizer", self.load_tokenizer)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @classmethod
    def setup_task(
        cls, cfg: HubertPretrainingConfig, **kwargs
    ) -> "HubertS2SPretrainingTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries
    
    def load_tokenizer(self):
        bpe_args = Namespace(**{'bpe': self.cfg.tokenizer_bpe_name, f"{self.cfg.tokenizer_bpe_name}_model": self.cfg.tokenizer_bpe_model})
        bpe_tokenizer = encoders.build_bpe(bpe_args)
        return bpe_tokenizer
    
    @property
    def s2s_tokenizer(self):
        return self.state.s2s_tokenizer

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        if not self.cfg.is_s2s:
            procs = [LabelEncoder(dictionary) for dictionary in dicts]
        else:
            logger.info(f"Using tokenizer")
            bpe_tokenizer = self.s2s_tokenizer
            procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dicts]
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]
        
        # hubert v1: pad_audio=True, random_crop=False;
        self.datasets[split] = HubertDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            is_s2s=self.cfg.is_s2s,
        )


    

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
    
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None, bos_token=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints, bos_token=bos_token
            )
    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.
        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """

        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer
            
            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )