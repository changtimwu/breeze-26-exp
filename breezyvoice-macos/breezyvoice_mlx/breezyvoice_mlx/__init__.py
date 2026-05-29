"""BreezyVoice-MLX — an Apple MLX port of BreezyVoice (CosyVoice2-derived zero-shot TTS).

Pipeline (mirrors the PyTorch source in ../BreezyVoice/cosyvoice):
    1. LLM        text -> speech tokens   (Conformer text encoder + RNN LM, AR loop)
    2. Flow       speech tokens -> mel     (conditional flow matching + UNet1D decoder)
    3. HiFiGAN    mel -> waveform          (HiFiGAN-NSF, F0-aware, weight_norm convs)

See PORTING_STATUS.md for the module-by-module port plan and difficulty map.
Research + divergence map: https://github.com/changtimwu/breeze-26-exp/issues/3
"""

__version__ = "0.0.1.dev0"
