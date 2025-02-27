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


from .deltalm import DeltaLMEncoder
from .transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
# from .transformer import (
#     TransformerModel,
#     TransformerDecoderBase,
#     TransformerEncoderBase,
# )

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
    deltalm_path: str = field(
        default=MISSING, metadata={"help": "path to lm model"}
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

class AVTNetEncoder(FairseqEncoder):
    def __init__(self, cfg, tgt_dict=None):
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task = tasks.setup_task(w2v_args.task)
        if state is not None:
            task.load_state_dict(state['task_state'])
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)
    
        self.w2v_model = model
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0


    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
        }
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # T x B X C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

@register_model("avtnet_delta_seq2seq", dataclass=AVTNetSeq2SeqConfig)
class AVTNetDeltamSeq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, tgt_dict, cfg):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        # load deltalm model
        deltalm_args = TransformerConfig()
        deltalm_args.pretrained_deltalm_checkpoint = cfg.deltalm_path
        deltalm_args = deltalm_base_encoder_architecture(deltalm_args)
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb
        deltalm_encoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        self.deltalm_encoder = DeltaLMEncoder(deltalm_args, tgt_dict, deltalm_encoder_embed_tokens)
        
        
        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        encoder = AVTNetEncoder(cfg,tgt_dict)
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        decoder = TransformerDecoder(cfg, tgt_dict, decoder_embed_tokens)

        return AVTNetDeltamSeq2Seq(encoder, decoder, tgt_dict, cfg)


    def forward(self, **kwargs):
        # 文本编码器的输出
        output = self.encoder(**kwargs)
        '''
        deltam_encoder_out {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }
        '''
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad():
            deltam_encoder_out = self.deltalm_encoder(src_tokens=kwargs["target"],src_lengths=kwargs["target_lengths"])
            deltam_encoder_out["encoder_out"] = deltam_encoder_out["encoder_out"][0]
            deltam_encoder_out["padding_mask"] = deltam_encoder_out["encoder_padding_mask"][0]
            deltam_encoder_out["encoder_embedding"] = deltam_encoder_out["encoder_embedding"][0]

        # av解码器的输出分别跟中文和英文做loss
        av_hubert_decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output)
        av_hubert_decoder_st_out = self.decoder(prev_output_tokens=kwargs['prev_st_output_tokens'], encoder_out=output)
        # 文本解码器的输出分别跟中文和英文做loss
        deltam_decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=deltam_encoder_out)
        deltam_decoder_st_out = self.decoder(prev_output_tokens=kwargs['prev_st_output_tokens'], encoder_out=deltam_encoder_out)
        
        return (
            av_hubert_decoder_out,
            av_hubert_decoder_st_out,
            deltam_decoder_out,
            deltam_decoder_st_out
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

def deltalm_base_encoder_architecture(args):
    
    args.encoder.embed_dim = 768
    args.encoder.ffn_embed_dim = 3072
    args.encoder.layers = 12
    args.encoder.attention_heads = 12
    args.encoder.normalize_before = False
    args.encoder.learned_pos = True
    args.max_source_positions = 500
    args.decoder.embed_dim = 768
    args.decoder.ffn_embed_dim = 3072
    args.decoder.layers = 6
    args.decoder.attention_heads = 12
    args.decoder.normalize_before = False
    args.decoder.learned_pos = True

    args.activation_fn = "gelu"
    args.no_scale_embedding = True
    args.layernorm_embedding = True
    args.max_positions = 512
    return args
    