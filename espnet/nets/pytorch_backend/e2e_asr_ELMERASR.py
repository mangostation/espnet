# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import math
from argparse import Namespace

import numpy
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import ErrorCalculator, end_detect
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD, Reporter
from espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (  # noqa: H301
    add_arguments_transformer_common,
)
from espnet.nets.pytorch_backend.transformer.attention import (  # noqa: H301
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.BertASR.decoder import Decoder
from espnet.nets.pytorch_backend.BertASR.MLM import BertForMaskedLMForBERTASR
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.LASO.encoder import Encoder
from espnet.nets.pytorch_backend.LASO.PDS import PDS
from espnet.nets.pytorch_backend.ELMERASR.ArgCondition import Argmax_Condition
from espnet.nets.pytorch_backend.ELMERASR.modeling_bert import *
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.BertASR.MLM import (  # noqa: H301
    LabelSmoothingCrossEntropy,
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask, target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.nets.pytorch_backend.LASO.embedding import PositionalEncoding

from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertTokenizerFast
from transformers.models.bert.modeling_bert import *


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group = add_arguments_transformer_common(group)

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return 4 * int(numpy.prod(self.subsample))

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        odim = 21128

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.adim = args.adim  # used for CTC (equal to d_model)
        self.mtlalpha = args.mtlalpha

        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        self.intermediate_ctc_weight = args.intermediate_ctc_weight
        self.intermediate_ctc_layers = None
        if args.intermediate_ctc_layer != "":
            self.intermediate_ctc_layers = [
                int(i) for i in args.intermediate_ctc_layer.split(",")
            ]

        self.encoder = None
        self.criterion = LabelSmoothingCrossEntropy()
        self.blank = 0
        self.sos = 101
        self.eos = 102
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()
        #PRETRAINED_MODEL_NAME = "bert-base-chinese"
        bertmodel = BertForMaskedLMForPermutation.from_pretrained("/work/m11115119/espnet/egs/aishell/asr1/bert_pretrain/bert_permutation/checkpoint-10000")
        self.decoder = bertmodel

        self.reset_parameters(args)

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None
        self.mix = torch.nn.Linear(768*2, 768)
        

        

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        enc_output = self.encoder(xs_pad, ilens, None)

        if ys_pad != None:
            # ys_in_pad, ys_out_pad = add_sos_eos(
            #     ys_pad, self.sos, self.eos, self.ignore_id
            # )
            ys_out_pad = ys_pad
            
            target_60 = torch.full((ys_out_pad.size(0),60), -1).to(ys_out_pad.device)
            target_60[:,:ys_out_pad.size(1)] = ys_out_pad
            maskn1 = target_60 == -1
            target_60[maskn1] = 0
            ys_out_pad = target_60
        # pred_pad = self.decoder(inputs_embeds=enc_output)

        # 
        # 
        # put ArgCon bert forward here
        
        #in bertMLM
        return_dict = True

        # outputs = self.decoder.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     exit_layers = exit_layers,
        # )

        #in bert model
        inputs_embeds = enc_output
        input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        device =  inputs_embeds.device
        past_key_values_length = 0
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        token_type_ids = None
        if token_type_ids is None:
            if hasattr(self.decoder.bert.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.decoder.bert.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.decoder.bert.get_extended_attention_mask(attention_mask, input_shape)

        encoder_extended_attention_mask = None

        head_mask = None
        head_mask = self.decoder.bert.get_head_mask(head_mask, self.decoder.bert.config.num_hidden_layers)

        embedding_output = self.decoder.bert.embeddings(
            input_ids=None,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        #in bert encoder

        hidden_states = embedding_output

        exit_hidden_states = torch.zeros(hidden_states.size()).to(hidden_states.device)
        has_all_exited = torch.zeros(hidden_states.shape[:-1]).unsqueeze(-1).bool().to(hidden_states.device)

        for i, layer_module in enumerate(self.decoder.bert.encoder.layer):

            if i > 0:
                
                # intermediate_logits = lm_head(hidden_states)  # using h_n to predict p_n
                # predicted_samples = intermediate_logits.argmax(dim=-1)
                arg_result = torch.argmax(self.decoder.cls(hidden_states),dim=-1)
                # predicted_embeds = self.embed_tokens(predicted_samples)
                predicted_embeds = self.decoder.bert.embeddings(input_ids=arg_result)
                if ys_pad != None:
                    # label_embeds = self.embed_tokens(labels)
                    label_embeds = self.decoder.bert.embeddings(input_ids=ys_out_pad)
                    probability = torch.ones(ys_out_pad.size()) * 0.3
                    control = torch.bernoulli(probability).unsqueeze(-1).to(ys_out_pad.device)
                    predicted_embeds = control * label_embeds + (1 - control) * predicted_embeds
                else:
                    arg_result = torch.argmax(self.decoder.cls(hidden_states),dim=-1)
                    predicted_embeds = self.decoder.bert.embeddings(input_ids=arg_result)
                hidden_states = concat_head(torch.cat([hidden_states, predicted_embeds], dim=-1))


            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                None,
                layer_head_mask,
                None,
                None,
                None,
                None,
            )

            hidden_states = layer_outputs[0]

            # hidden_states = self.mix(torch.cat((self.embed(self.argcon(hidden_states)), hidden_states), -1))

            # arg_result = torch.argmax(self.decoder.cls(hidden_states),dim=-1)

            # condition_result = self.decoder.bert.embeddings(input_ids=arg_result)

            # hidden_states = self.mix(torch.cat((condition_result, hidden_states),-1))
            

            # if exit_layers is None:
            #     raise ValueError("The exit layers should be provided!")
            # else:
            #     # whether a word exit in the current layer, True: exit, False: not exit
            #     # if True, copy the current hidden state to the cache (used in later layers)
            #     exited_signal = torch.eq(exit_layers, i).unsqueeze(-1)
            #     exit_hidden_states = torch.where(exited_signal, hidden_states, exit_hidden_states)

            #     # if all words have exited, then break
            #     has_all_exited = has_all_exited | exited_signal
            #     if torch.all(has_all_exited):
            #         break

            #     # only copy hidden states to the layer higher than the exit layer
            #     copy_signal = torch.le(exit_layers, i).unsqueeze(-1)
            #     hidden_states = torch.where(copy_signal, exit_hidden_states.detach(), hidden_states)

        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

        #out bert encoder
        sequence_output = encoder_outputs[0]
        #pooled_output = self.decoder.bert.pooler(sequence_output)

        bert_output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=None,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

        #out bert model

        sequence_output = bert_output[0]
        prediction_scores = self.decoder.cls(sequence_output)

        pred_pad = MaskedLMOutput(
            loss=None,
            logits=prediction_scores,
            hidden_states=bert_output.hidden_states,
            attentions=bert_output.attentions,
        ).logits
        #out bertMLM
        # 
        # 

        self.pred_pad = pred_pad

        if ys_pad != None:
            # ys_in_pad, ys_out_pad = add_sos_eos(
            #     ys_pad, self.sos, self.eos, self.ignore_id
            # )
            # ys_out_pad = ys_pad
            
            # target_60 = torch.full((ys_out_pad.size(0),60), -1).to(ys_out_pad.device)
            # target_60[:,:ys_out_pad.size(1)] = ys_out_pad
            # maskn1 = target_60 == -1
            # target_60[maskn1] = 0
            # ys_out_pad = target_60

            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = 0
            # self.acc = th_accuracy(
            #     pred_pad.view(-1, self.odim), ys_pad, ignore_label=self.ignore_id
            # )

            # 2. forward decoder
            # if self.decoder is not None:
            #     ys_in_pad, ys_out_pad = add_sos_eos(
            #         ys_pad, self.sos, self.eos, self.ignore_id
            #     )
            #     ys_mask = target_mask(ys_in_pad, self.ignore_id)
            #     pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            #     self.pred_pad = pred_pad

            #     # 3. compute attention loss
            #     loss_att = self.criterion(pred_pad, ys_out_pad)
            #     self.acc = th_accuracy(
            #         pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            #     )
            # else:
            #     loss_att = None
            #     self.acc = None

            # TODO(karita) show predicted text
            # TODO(karita) calculate these stats
            cer_ctc = None
            loss_intermediate_ctc = 0.0
            if self.mtlalpha == 0.0:
                loss_ctc = None
            else:
                batch_size = xs_pad.size(0)
                hs_len = hs_mask.view(batch_size, -1).sum(1)
                loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
                if not self.training and self.error_calculator is not None:
                    ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                    cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
                # for visualization
                if not self.training:
                    self.ctc.softmax(hs_pad)

                if self.intermediate_ctc_weight > 0 and self.intermediate_ctc_layers:
                    for hs_intermediate in hs_intermediates:
                        # assuming hs_intermediates and hs_pad has same length / padding
                        loss_inter = self.ctc(
                            hs_intermediate.view(batch_size, -1, self.adim), hs_len, ys_pad
                        )
                        loss_intermediate_ctc += loss_inter

                    loss_intermediate_ctc /= len(self.intermediate_ctc_layers)

            # 5. compute cer/wer
            if self.training or self.error_calculator is None or self.decoder is None:
                cer, wer = None, None
            else:
                ys_hat = pred_pad.argmax(dim=-1)
                cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

            # copied from e2e_asr
            alpha = self.mtlalpha
            if alpha == 0:
                self.loss = loss_att
                loss_att_data = float(loss_att)
                loss_ctc_data = None
            elif alpha == 1:
                self.loss = loss_ctc
                if self.intermediate_ctc_weight > 0:
                    self.loss = (
                        1 - self.intermediate_ctc_weight
                    ) * loss_ctc + self.intermediate_ctc_weight * loss_intermediate_ctc
                loss_att_data = None
                loss_ctc_data = float(loss_ctc)
            else:
                self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
                if self.intermediate_ctc_weight > 0:
                    self.loss = (
                        (1 - alpha - self.intermediate_ctc_weight) * loss_att
                        + alpha * loss_ctc
                        + self.intermediate_ctc_weight * loss_intermediate_ctc
                    )
                loss_att_data = float(loss_att)
                loss_ctc_data = float(loss_ctc)

            loss_data = float(self.loss)
            if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
                self.reporter.report(
                    loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
                )
            else:
                logging.warning("loss (=%f) is not correct", loss_data)
            return self.loss
        else:
            
            return pred_pad

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        return self.encoder.encode(x)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0)
        enc_output = self.decoder(inputs_embeds=enc_output)

        if self.mtlalpha == 1.0:
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            lpz = self.ctc.argmax(enc_output)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps
        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        #h = enc_output.squeeze(0)
        enc_output = enc_output.view(-1, self.odim)
        #print(enc_output)
        h = torch.argmax(torch.log_softmax(enc_output, dim=1),dim=1)
        #print(h)
        #print(h)
        mask = h == 102
        ylen = torch.nonzero(mask).squeeze()
        #print(h[:ylen[0]])
        #print(ylen[0])

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        print(ylen.ndim)
        if ylen.ndim == 0:
            hyp["yseq"] = h[:ylen].tolist()
        else:
            hyp["yseq"] = h[:ylen[0]].tolist()
        hyps = [hyp]
        ended_hyps = []

        #print(hyp)

        """
        traced_decoder = None
        
        for i in range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]
        
        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        
        return nbest_hyps
        """
        return hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
