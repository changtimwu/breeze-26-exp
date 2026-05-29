"""Flow matching wrapper (speech tokens -> mel).  [PORT STATUS: STUB — MEDIUM]

PyTorch source: ../BreezyVoice/cosyvoice/flow/flow.py (141 lines)
Class: MaskedDiffWithXvec (encoder -> encoder_proj -> length_regulator -> CFM decoder)

Port the inference() path (source lines 99-141):
  * xvec: F.normalize + spk_embed_affine_layer
  * concat(prompt_token, token); input_embedding; mask
  * encoder (Conformer) -> encoder_proj
  * feat_len = (token_len / 50 * 22050 / 256).int()
  * length_regulator(h, feat_len)
  * build conds from prompt_feat, decoder(mu, mask, spks, cond, n_timesteps=10)
  * trim the prompt-feat prefix off the output
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MaskedDiffWithXvec(nn.Module):  # TODO: implement
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("MaskedDiffWithXvec MLX port pending.")

    def inference(self, token, token_len, prompt_token, prompt_token_len,
                  prompt_feat, prompt_feat_len, embedding) -> mx.array:  # pragma: no cover
        raise NotImplementedError
