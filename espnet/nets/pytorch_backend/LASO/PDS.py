from espnet.nets.pytorch_backend.LASO.PDS_layer import PDSLayer
from espnet.nets.pytorch_backend.LASO.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.LASO.attention import MultiHeadedAttention
from torch import nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.LASO.positionwise_feed_forward import (
    PositionwiseFeedForward,
)

class PDS(nn.Module):
    def __init__(
        self,
        num_blocks=4,
        attention_dim=512,
        attention_heads=8,
        linear_units=2048,
        dropout_rate=0.1,
        src_attention_dropout_rate=0.1,
        norm=None
    ):
        super(PDS, self).__init__()
        self.decoders = repeat(
            num_blocks-1,
            lambda lnum: PDSLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate, F.glu),
                dropout_rate,
                True,
            ),
        )
        self.PDS_input = PDSLayer(
            attention_dim,
            MultiHeadedAttention(
                attention_heads, attention_dim, src_attention_dropout_rate
            ),
            PositionwiseFeedForward(attention_dim, linear_units, dropout_rate, F.glu),
            dropout_rate,
            False,
        )
    
    def forward(self, tgt, memory, memory_mask):
        
        tgt, tgt_mask, memory, memory_mask = self.PDS_input(
            tgt, None, memory, memory_mask
        )
        
        tgt, tgt_mask, memory, memory_mask = self.decoders(
            tgt, None, memory, memory_mask
        )

        return tgt, tgt_mask