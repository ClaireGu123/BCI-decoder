
from typing import Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .utils import ModelOutput

from models.activations import ACT2FN

# model outputs
@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class DeBCIBaseModelOutput(ModelOutput):
    """
    Base class for models that have been trained with the DeBCIBaseModelOutput loss objective.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    extract_features: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DeBCINoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.model.conv_dim[layer_id - 1] if layer_id > 0 else config.model.in_channels
        self.out_conv_dim = config.model.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.model.conv_kernel[layer_id],
            stride=config.model.conv_stride[layer_id],
            bias=config.model.conv_bias,
        )
        self.activation = ACT2FN[config.model.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states
    
class DeBCILayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.model.conv_dim[layer_id - 1] if layer_id > 0 else config.model.in_channels
        self.out_conv_dim = config.model.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.model.conv_kernel[layer_id],
            stride=config.model.conv_stride[layer_id],
            bias=config.model.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.model.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states
    
class DeBCIFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.model.conv_dim[-1], eps=config.model.layer_norm_eps)
        self.projection = nn.Linear(config.model.conv_dim[-1], config.model.hidden_size)
        self.dropout = nn.Dropout(config.model.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states
    
class DeBCISamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states
    
class DeBCIPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.model.hidden_size,
            config.model.hidden_size,
            kernel_size=config.model.num_conv_pos_embeddings,
            padding=config.model.num_conv_pos_embeddings // 2,
            groups=config.model.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm


        self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = DeBCISamePadLayer(config.model.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.model.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->DeBCI
class DeBCIAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        # #######Falsh Attention#####################
        # torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, 
        #     dropout_p=self.dropout if self.training else 0, is_causal=True)
        #############################################

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class DeBCIFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.model.activation_dropout)

        self.intermediate_dense = nn.Linear(config.model.hidden_size, config.model.intermediate_size)
        if isinstance(config.model.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.model.hidden_act]
        else:
            self.intermediate_act_fn = config.model.hidden_act

        self.output_dense = nn.Linear(config.model.intermediate_size, config.model.hidden_size)
        self.output_dropout = nn.Dropout(config.model.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states
    
class DeBCIFeatureEncoder(nn.Module):

    """ #TODO: checkout the line 436
    (place holder) Feature extractor for the neural decoder model"""
    def __init__(self, config):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            *[DeBCINoLayerNormConvLayer(config, layer_id=layer) 
             for layer in range(config.model.n_conv_layers)]
        )
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, x):
        return self.feature_encoder(x.permute(0, 2, 1))
    
class DeBCIEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DeBCIAttention(
            embed_dim=config.model.hidden_size,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.model.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.model.hidden_size, eps=config.model.layer_norm_eps)
        self.feed_forward = DeBCIFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.model.hidden_size, eps=config.model.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs    

    
class DeBCIEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = DeBCIPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.model.hidden_size, eps=config.model.layer_norm_eps)
        self.dropout = nn.Dropout(config.model.hidden_dropout)
        self.layers = nn.ModuleList([DeBCIEncoderLayer(config) for _ in range(config.model.num_hidden_layers)])
        self.gradient_checkpointing = False
        self._requires_grad = True

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)


        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.model.layerdrop) else False

            layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
            hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
class DeBCIModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = DeBCIFeatureEncoder(config)
        self.feature_projection = DeBCIFeatureProjection(config)
        self.encoder = DeBCIEncoder(config)
        self.gradient_checkpointing = False
        self._requires_grad = True
        self.adapter = None
        


    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()


    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DeBCIBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.model.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.model.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.model.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        # hidden_states = self._mask_hidden_states(
        #     hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        # ) # No masking is happenning here

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return DeBCIBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
    
class DeBCIForCTC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_model = DeBCIModel(config)
        self.class_projector = nn.Linear(config.model.hidden_size, 
                                         config.model.n_classes)
    def forward(self,hidden_states):
        hidden_states = self.base_model(hidden_states)[0] # 0 is the hidden states, 1 is the extracted feature
        return self.class_projector(hidden_states)