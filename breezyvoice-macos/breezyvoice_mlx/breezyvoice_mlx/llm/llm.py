"""Text-to-speech-token LLM.  [PORT STATUS: IMPLEMENTED — parity-tested]

MLX port of TransformerLM from ../BreezyVoice/cosyvoice/llm/llm.py — BreezyVoice's
CosyVoice-v1 LLM (Conformer text encoder + Transformer LM backbone + AR decode).
NOT Qwen2 (see PORTING_STATUS.md).

Pipeline of inference(): concat(prompt_text, text) -> text_embedding -> Conformer
encode -> affine; build lm_input = [sos_eos, spk_emb, text, task_id, prompt_speech]
then greedily/sampled-decode speech tokens via the cached TransformerEncoder
forward_chunk loop, stopping at the EOS id (== speech_token_size).

Weight names match the torch module (text_embedding, text_encoder.*,
text_encoder_affine_layer, llm_embedding, llm.*, llm_decoder, speech_embedding,
spk_embed_affine_layer) so a converted checkpoint loads directly.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..transformer.encoder import ConformerEncoder, TransformerEncoder


def top_k_sampling(logp: mx.array, k: int = 25, key: Optional[mx.array] = None) -> int:
    """Top-k multinomial sample over a (V,) log-prob vector. Mirrors the torch
    sampling_ids (softmax->topk->multinomial). `logp` is already log-softmax."""
    probs = mx.exp(logp)
    idx = mx.argpartition(-probs, k)[:k]
    top = probs[idx]
    top = top / mx.sum(top)
    choice = mx.random.categorical(mx.log(top), key=key)
    return int(idx[choice].item())


def greedy_sampling(logp: mx.array, k: int = 25, key: Optional[mx.array] = None) -> int:
    return int(mx.argmax(logp).item())


class TransformerLM(nn.Module):
    def __init__(self, text_encoder_input_size: int, llm_input_size: int,
                 llm_output_size: int, text_token_size: int, speech_token_size: int,
                 text_encoder: ConformerEncoder, llm: TransformerEncoder,
                 spk_embed_dim: int = 192):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        self.sos_eos = 0
        self.task_id = 1

        self.text_embedding = nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(text_encoder.output_size(),
                                                    llm_input_size)
        self.llm_embedding = nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.speech_embedding = nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, llm_input_size)

    def encode(self, text: mx.array, text_lengths: mx.array):
        out, mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1,
                                      num_decoding_left_chunks=-1)
        out_lens = mx.sum(mask.reshape(mask.shape[0], -1), axis=1)
        out = self.text_encoder_affine_layer(out)
        return out, out_lens

    def prompt_logits(self, text: mx.array, text_len: mx.array,
                      prompt_text: mx.array, prompt_text_len: mx.array,
                      prompt_speech_token: mx.array, prompt_speech_token_len: mx.array,
                      embedding: mx.array) -> mx.array:
        """Deterministic: build the AR prompt and return the LLM's per-position
        logits over it (1, T_prompt, speech_token_size+1). Used for quantization
        fidelity A/B (greedy decode degenerates this model, so compare logits)."""
        text = mx.concatenate([prompt_text, text], axis=1)
        text_len = text_len + prompt_text_len
        text = self.text_embedding(text)
        text, text_len = self.encode(text, text_len)
        if embedding.shape[0] != 0:
            embedding = embedding / mx.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = mx.expand_dims(self.spk_embed_affine_layer(embedding), axis=1)
        else:
            embedding = mx.zeros((1, 0, self.llm_input_size))
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        prompt_emb = (self.speech_embedding(prompt_speech_token)
                      if int(prompt_speech_token_len.item()) != 0
                      else mx.zeros((1, 0, self.llm_input_size)))
        lm_input = mx.concatenate([sos_eos_emb, embedding, text, task_id_emb, prompt_emb], axis=1)
        n = lm_input.shape[1]
        y_pred, _ = self.llm.forward_chunk(lm_input, offset=0, required_cache_size=-1,
                                           att_cache=None, att_mask=mx.tril(mx.ones((1, n, n))))
        return self.llm_decoder(y_pred)

    def inference(self, text: mx.array, text_len: mx.array,
                  prompt_text: mx.array, prompt_text_len: mx.array,
                  prompt_speech_token: mx.array, prompt_speech_token_len: mx.array,
                  embedding: mx.array, sampling: int = 25,
                  max_token_text_ratio: float = 30, min_token_text_ratio: float = 3,
                  sampling_fn: Callable = top_k_sampling) -> mx.array:
        text = mx.concatenate([prompt_text, text], axis=1)
        text_len = text_len + prompt_text_len
        text = self.text_embedding(text)
        text, text_len = self.encode(text, text_len)

        if embedding.shape[0] != 0:
            embedding = embedding / mx.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = mx.expand_dims(embedding, axis=1)
        else:
            embedding = mx.zeros((1, 0, self.llm_input_size))

        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if int(prompt_speech_token_len.item()) != 0:
            prompt_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_emb = mx.zeros((1, 0, self.llm_input_size))
        lm_input = mx.concatenate([sos_eos_emb, embedding, text, task_id_emb,
                                   prompt_emb], axis=1)

        text_only_len = float((text_len - prompt_text_len).item())
        min_len = int(text_only_len * min_token_text_ratio)
        max_len = int(text_only_len * max_token_text_ratio)

        out_tokens: List[int] = []
        att_cache = None
        for i in range(max_len):
            n = lm_input.shape[1]
            att_mask = mx.tril(mx.ones((1, n, n)))
            y_pred, att_cache = self.llm.forward_chunk(
                lm_input, offset=0, required_cache_size=-1,
                att_cache=att_cache, att_mask=att_mask)
            logits = self.llm_decoder(y_pred[:, -1])
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            logp = logp.reshape(-1)
            if i < min_len:                       # forbid EOS before min length
                logp = mx.concatenate([logp[:-1], mx.array([-float("inf")])])
            top_id = sampling_fn(logp, sampling)
            if top_id == self.speech_token_size:  # EOS
                break
            out_tokens.append(top_id)
            lm_input = self.speech_embedding.weight[top_id].reshape(1, 1, -1)

        return mx.array([out_tokens], dtype=mx.int64)
