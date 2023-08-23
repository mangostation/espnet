from transformers.models.bert.modeling_bert import *
import torch.nn as nn
import torch.nn.functional as F


def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        #print(log_preds.shape)
        #print(target.shape)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)





class BertForMaskedLMForBERTASR(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = outputs[0]

        sequence_output = torch.add(sequence_output, inputs_embeds)

        #print(sequence_output.shape)
        prediction_scores = self.cls(sequence_output)
        #print(prediction_scores.shape)

        return prediction_scores
        # masked_lm_loss = None
        # if labels is not None:
        #     loss_fct = LabelSmoothingCrossEntropy()  # -100 index = padding token
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # if not return_dict:
        #     output = (prediction_scores,) + outputs[2:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # return MaskedLMOutput(
        #     loss=masked_lm_loss,
        #     logits=prediction_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )