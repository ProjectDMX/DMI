# Auto-generated spec from: modeling_qwen3.py
# Add H("name", variable) where you want hooks, then run: dmi compile

from monitoring.compiler.dsl import H, spec

@spec(source="modeling_qwen3.py")
class Qwen3RMSNorm:
    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states * torch.rsqrt(...)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3MLP:
    def forward(self, x):
        down_proj = self.down_proj(...)
        return down_proj


class Qwen3Attention:
    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor], past_key_values: Optional[Cache]=None, cache_position: Optional[torch.LongTensor]=None, **kwargs: Unpack[FlashAttentionKwargs]):
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(...)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(...)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(...)
        (query_states, key_states) = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        (attn_output, attn_weights) = attention_interface(self, query_states, key_states, value_states, attention_mask)
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)


class Qwen3DecoderLayer:
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, cache_position: Optional[torch.LongTensor]=None, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]]=None, **kwargs: Unpack[TransformersKwargs]):
        hidden_states = self.input_layernorm(hidden_states)
        (hidden_states, _) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        H("mlp_out", hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3RotaryEmbedding:
    def forward(self, x, position_ids):
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))


class Qwen3Model:
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Cache]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None, **kwargs: Unpack[TransformersKwargs]):
        inputs_embeds = self.embed_tokens(input_ids)
        past_key_values = DynamicCache()
        causal_mask_mapping['sliding_attention'] = create_sliding_window_causal_mask()
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None)


class Qwen3ForCausalLM:
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Cache]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None, logits_to_keep: Union[int, torch.Tensor]=0, **kwargs: Unpack[TransformersKwargs]):
        logits = self.lm_head(...)
        loss = self.loss_function(logits=logits, labels=labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
