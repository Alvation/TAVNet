#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from dataclasses import dataclass, field, is_dataclass
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    scoring: str = "sacrebleu"
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )



def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("hybrid.speech_translate")

    utils.import_user_module(cfg.common)
    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path])
    models = [model.eval().cuda() for model in models]
    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)

    tokenizer = task.build_tokenizer(saved_cfg.tokenizer)
    bpe = task.build_bpe(saved_cfg.bpe)


    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu


    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    
    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    
    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    
    cfg.generation.score_reference = False  #
    save_attention_plot = cfg.generation.print_alignment is not None
    cfg.generation.print_alignment = None  #
    generator = task.build_generator(
        models, cfg.generation,  extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    
    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        
        # prefix_tokens = (torch.LongTensor([tgt_dict.index("</st_s>")]),0)
        prefix_tokens = None
 
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target_st"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
            bos_token=tgt_dict.index("</st_s>")
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target_st"] is not None

            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target_st"][i, :], tgt_dict.pad()).int().cpu()
                )

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                    sample_id
                )
                target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        cfg.common_eval.post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not cfg.common_eval.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                    if cfg.generation.print_alignment == "hard":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )
                    if cfg.generation.print_alignment == "soft":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [",".join(src_probs) for src_probs in alignment]
                                ),
                            ),
                            file=output_file,
                        )

                    if cfg.generation.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if cfg.generation.retain_iter_history:
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )

                # Score only the top hypothesis
                if has_target and j == 0:
                    if (
                        align_dict is not None
                        or cfg.common_eval.post_process is not None
                    ):
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )
                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {:,} sentences ({:,} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    if has_target:
        if saved_cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            "Generate {} with beam={}: {}".format(
                cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string()
            ),
            file=output_file,
        )

    return scorer

@hydra.main(config_path=config_path, config_name="infer_st")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))
    return

def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()