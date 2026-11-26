"""EnCodec FLOP counting copied verbatim from the integration notebook 08 (scripts/flops.py)."""
import torch
import torch.nn as nn


def _count_neural_flops(module: nn.Module, sample_input: torch.Tensor) -> tuple[float, float]:
    module.eval()
    with torch.no_grad():
        try:
            from thop import profile
            macs, params = profile(module, inputs=(sample_input,), verbose=False)
            return float(macs * 2.0), float(params)
        except ImportError:
            pass

        try:
            from ptflops import get_model_complexity_info
            input_shape = tuple(sample_input.shape[1:])
            macs, params = get_model_complexity_info(
                module, input_shape, as_strings=False, print_per_layer_stat=False, verbose=False,
            )
            return float(macs * 2.0), float(params)
        except ImportError:
            pass

        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            _ = module(sample_input)
        params = sum(p.numel() for p in module.parameters())
        return float(fc.get_total_flops()), float(params)


class _EncodecEncoderWrapper(nn.Module):
    def __init__(self, encodec_model, bandwidth: float):
        super().__init__()
        self.model = encodec_model
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor):
        return self.model.encode(x, bandwidth=self.bandwidth)


class _EncodecDecoderWrapper(nn.Module):
    def __init__(self, encodec_model, audio_scales):
        super().__init__()
        self.model = encodec_model
        self.audio_scales = audio_scales

    def forward(self, audio_codes: torch.Tensor):
        return self.model.decode(audio_codes, self.audio_scales, None)
