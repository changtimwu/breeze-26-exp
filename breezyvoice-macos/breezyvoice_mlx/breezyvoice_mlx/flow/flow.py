"""Flow matching orchestrator (speech tokens -> mel).  [PORT STATUS: IMPLEMENTED — parity-tested]

MLX port of MaskedDiffWithXvec from ../BreezyVoice/cosyvoice/flow/flow.py.
Wires the verified pieces: input_embedding -> ConformerEncoder -> encoder_proj ->
InterpolateRegulator -> ConditionalCFM(estimator=ConditionalDecoder).

inference() mirrors the torch source: prepend the prompt token/feat, encode,
length-regulate to mel frames (token_len/50*22050/256), build the prompt-feat
conditioning, run the CFM, and trim the prompt-feat prefix off the output.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..transformer.encoder import ConformerEncoder, make_pad_mask
from .length_regulator import InterpolateRegulator
from .flow_matching import ConditionalCFM


class MaskedDiffWithXvec(nn.Module):
    def __init__(self, input_size: int = 512, output_size: int = 80,
                 spk_embed_dim: int = 192, vocab_size: int = 4096,
                 input_frame_rate: int = 50, encoder: ConformerEncoder = None,
                 length_regulator: InterpolateRegulator = None,
                 decoder: ConditionalCFM = None):
        super().__init__()
        self.output_size = output_size
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(encoder.output_size(), output_size)
        self.length_regulator = length_regulator
        self.decoder = decoder

    def inference(self, token, token_len, prompt_token, prompt_token_len,
                  prompt_feat, prompt_feat_len, embedding,
                  n_timesteps: int = 10, z: mx.array = None) -> mx.array:
        # speaker embedding
        embedding = embedding / mx.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = self.spk_embed_affine_layer(embedding)

        # prepend prompt tokens
        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len
        mask = mx.expand_dims((~make_pad_mask(token_len, token.shape[1])).astype(mx.float32), -1)
        token = self.input_embedding(mx.clip(token, 0, None)) * mask

        # encode + project + length-regulate to mel frames
        h, _ = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        feat_len = (token_len.astype(mx.float32) / self.input_frame_rate * 22050 / 256).astype(mx.int32)
        h, _ = self.length_regulator(h, feat_len)

        # prompt-feat conditioning (B=1)
        out_len = int(mx.max(feat_len).item())
        conds = mx.zeros((1, out_len, self.output_size))
        pf = int(prompt_feat.shape[1])
        if pf != 0:
            conds = mx.concatenate([prompt_feat, conds[:, pf:]], axis=1)
        conds = mx.swapaxes(conds, 1, 2)                    # (B, C, T)

        feat_mask = mx.expand_dims((~make_pad_mask(feat_len, out_len)).astype(mx.float32), 1)  # (B,1,T)
        feat = self.decoder(mu=mx.swapaxes(h, 1, 2), mask=feat_mask, spks=embedding,
                            cond=conds, n_timesteps=n_timesteps, z=z)   # (B, C, T)
        if pf != 0:
            feat = feat[:, :, pf:]
        return feat
