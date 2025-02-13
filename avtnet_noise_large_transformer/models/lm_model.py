# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,logging
import contextlib
import tempfile
from argparse import Namespace
from typing import Any, Optional

import argparse
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass,ChoiceEnum
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING
from fairseq.modules import LayerNorm
from fairseq.models.transformer import TransformerEncoder

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from fairseq.models.av_hubert import AVHubertModel
    from decoder import TransformerDecoder
else:
    from fairseq.models.av_hubert import AVHubertModel
    from .decoder import TransformerDecoder

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])
LAYER_TYPE_CHOICES = ChoiceEnum(["transformer", "conformer", "trf_adp"])

@dataclass
class AVTNetAsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None

@dataclass
class AVTNetSeq2SeqConfig(AVTNetAsrConfig):
    # deocder 
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})


@register_model("t_model", dataclass=AVTNetSeq2SeqConfig)
class AVTNetTransformerSeq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, tgt_dict, cfg):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        lm_args = Namespace()
        lm_args = transformer_base_architecture(lm_args)
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb
        lm_encoder_embed_tokens = build_embedding(tgt_dict, lm_args.encoder_embed_dim)
        encoder = TransformerEncoder(lm_args,tgt_dict,lm_encoder_embed_tokens)
        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        decoder = TransformerDecoder(cfg, tgt_dict, decoder_embed_tokens)
        
        return AVTNetTransformerSeq2Seq(encoder, decoder, tgt_dict, cfg)


    def forward(self, **kwargs):
        lm_encoder_out = self.encoder(src_tokens=kwargs["target"],src_lengths=kwargs["target_lengths"])
        lm_encoder_out["encoder_out"] = lm_encoder_out["encoder_out"][0]
        lm_encoder_out["padding_mask"] = lm_encoder_out["encoder_padding_mask"][0]
        lm_encoder_out["encoder_embedding"] = lm_encoder_out["encoder_embedding"][0]

        lm_decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=lm_encoder_out)
        lm_decoder_st_out = self.decoder(prev_output_tokens=kwargs['prev_st_output_tokens'], encoder_out=lm_encoder_out)
        return (
            lm_decoder_out,
            lm_decoder_st_out
        )

    def get_st_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target_st"]
    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def transformer_base_architecture(args):
    args.max_source_positions = 1024
    args.encoder_embed_path = None
    args.encoder_embed_dim = 1024
    args.encoder_ffn_embed_dim = 3072
    args.encoder_layers = 6
    args.encoder_attention_heads = 8
    args.encoder_normalize_before = False
    args.encoder_learned_pos = False
    args.attention_dropout = 0.0
    args.activation_dropout = 0.0
    args.activation_fn = "relu"
    args.dropout = 0.1
    args.final_dropout = 0.4
    args.adaptive_softmax_cutoff = None
    args.adaptive_softmax_dropout = 0
    args.share_all_embeddings = False
    args.no_token_positional_embeddings = False
    args.adaptive_input = False
    args.no_cross_attention = False
    args.cross_self_attention = False
    args.no_scale_embedding = False
    args.layernorm_embedding = False
    args.tie_adaptive_weights = False
    args.checkpoint_activations = False
    args.offload_activations = False
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = None
    args.encoder_layerdrop = 0
    args.quant_noise_pq = 0
    args.quant_noise_pq_block_size = 8
    args.quant_noise_scalar = 0
    return args