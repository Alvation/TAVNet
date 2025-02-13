# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.data.data_utils import lengths_to_mask
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss



@register_criterion(
    "avtnet_delta_criterion",
    dataclass=LabelSmoothedCrossEntropyCriterionConfig,
)
class AVTNetCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )


    def get_st_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_st_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)
    

    def compute_st_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_st_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
    
    def compute_st_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_st_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])

        loss, nll_loss = self.compute_loss(model, net_output[0], sample, reduce=reduce)
        st_loss, st_null_loss = self.compute_st_loss(model, net_output[1], sample, reduce=reduce)
        delta_loss, delta_null_loss = self.compute_loss(model, net_output[2], sample, reduce=reduce)
        delta_st_loss, delta_st_null_loss = self.compute_st_loss(model, net_output[3], sample, reduce=reduce)

        # deltalm_cosine = 1 / (torch.nn.functional.cosine_similarity(net_output[0][0].view(-1, net_output[2][0].size(-1)) ,net_output[2][0].view(-1, net_output[0][0].size(-1))).mean()+0.001)
        # deltalm_cosine_st = 1 / (torch.nn.functional.cosine_similarity(net_output[1][0].view(-1, net_output[2][0].size(-1)) ,net_output[3][0].view(-1, net_output[0][0].size(-1))).mean()+0.001)


        loss = loss + st_loss + delta_loss + delta_st_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        sample_size_st = (
            sample["target_st"].size(0) if self.sentence_avg else sample["ntokens_st"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),

            "delta_loss": utils.item(delta_loss.data),
            "delta_null_loss": utils.item(delta_null_loss.data),
            
            "st_loss": utils.item(st_loss.data),
            "st_null_loss": utils.item(st_null_loss.data),

            "delta_st_loss": utils.item(delta_st_loss.data),
            "delta_st_null_loss": utils.item(delta_st_null_loss.data),
            
            # "deltalm_cosine": utils.item(deltalm_cosine.data),
            # "deltalm_cosine_st": utils.item(deltalm_cosine_st.data),
            
            "ntokens": sample["ntokens"],
            "ntokens_st": sample["ntokens_st"],
            
            "nsentences": sample["target"].size(0),
            "nsentences_st": sample["target_st"].size(0),

            "sample_size": sample_size,
            "sample_size_st": sample_size_st,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output[0], sample)
            n_correct_st, total_st = self.compute_st_accuracy(model, net_output[1], sample)
            n_correct_deltalm, total_deltalm = self.compute_accuracy(model, net_output[2], sample)
            n_correct_deltalm_st, total_deltalm_st = self.compute_st_accuracy(model, net_output[3], sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

            logging_output["n_correct_st"] = utils.item(n_correct_st.data)
            logging_output["total_st"] = utils.item(total_st.data)

            logging_output["n_correct_deltalm"] = utils.item(n_correct_deltalm.data)
            logging_output["total_deltalm"] = utils.item(total_deltalm.data)
            
            logging_output["n_correct_deltalm_st"] = utils.item(n_correct_deltalm_st.data)
            logging_output["total_deltalm_st"] = utils.item(total_deltalm_st.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        st_null_loss_sum = sum(log.get("st_null_loss", 0) for log in logging_outputs)

        delta_loss_sum = sum(log.get("delta_loss", 0) for log in logging_outputs)
        delta_null_loss_sum = sum(log.get("delta_null_loss", 0) for log in logging_outputs)

        delta_st_loss_sum = sum(log.get("delta_st_loss", 0) for log in logging_outputs)
        delta_st_null_loss_sum = sum(log.get("delta_st_null_loss", 0) for log in logging_outputs)

        sample_size_st = sum(log.get("sample_size_st", 0) for log in logging_outputs)
        
        ntokens_st = sum(log.get("ntokens_st", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        
        metrics.log_derived(
            "ppl_st", lambda meters: utils.get_perplexity(meters["st_null_loss"].avg)
        )

        metrics.log_scalar(
            "st_loss", st_loss_sum / ntokens_st / math.log(2), sample_size_st, round=3
        )
        metrics.log_scalar(
            "st_null_loss", st_null_loss_sum / ntokens_st / math.log(2), sample_size_st, round=3
        )


        metrics.log_scalar(
            "delta_loss", delta_loss_sum / ntokens / math.log(2), sample_size_st, round=3
        )
        metrics.log_scalar(
            "delta_null_loss", delta_null_loss_sum / ntokens / math.log(2), sample_size_st, round=3
        )

        metrics.log_scalar(
            "delta_st_loss", delta_st_loss_sum / ntokens_st / math.log(2), sample_size_st, round=3
        )
        metrics.log_scalar(
            "delta_st_null_loss", delta_st_null_loss_sum / ntokens_st / math.log(2), sample_size_st, round=3
        )

        total_st = utils.item(sum(log.get("total_st", 0) for log in logging_outputs))
        if total_st > 0:
            metrics.log_scalar("total_st", total_st)
            n_correct_st = utils.item(
                sum(log.get("n_correct_st", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct_st", n_correct_st)
            metrics.log_derived(
                "accuracy_st",
                lambda meters: round(
                    meters["n_correct_st"].sum * 100.0 / meters["total_st"].sum, 3
                )
                if meters["total_st"].sum > 0
                else float("nan"),
            )
        
        total_deltalm = utils.item(sum(log.get("total_deltalm", 0) for log in logging_outputs))
        if total_deltalm > 0:
            metrics.log_scalar("total_deltalm", total_deltalm)
            n_correct_deltalm = utils.item(
                sum(log.get("n_correct_deltalm", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct_deltalm", n_correct_deltalm)
            metrics.log_derived(
                "accuracy_deltalm",
                lambda meters: round(
                    meters["n_correct_deltalm"].sum * 100.0 / meters["total_deltalm"].sum, 3
                )
                if meters["total_deltalm"].sum > 0
                else float("nan"),
            )
        
        total_deltalm_st = utils.item(sum(log.get("total_deltalm_st", 0) for log in logging_outputs))
        if total_deltalm_st > 0:
            metrics.log_scalar("total_deltalm_st", total_st)
            n_correct_deltalm_st = utils.item(
                sum(log.get("n_correct_deltalm_st", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct_deltalm_st", n_correct_deltalm_st)
            metrics.log_derived(
                "accuracy_deltalm_st",
                lambda meters: round(
                    meters["n_correct_deltalm_st"].sum * 100.0 / meters["total_deltalm_st"].sum, 3
                )
                if meters["total_deltalm_st"].sum > 0
                else float("nan"),
            )