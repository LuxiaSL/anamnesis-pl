"""Opt-in eager replay lane with per-layer attention reduction.

Supports the dense Llama probe-free full battery. The NumPy reference and its
API are unchanged. An emitted receipt is NOT a certification: the external
fidelity harness must pass on the declared reference cohort first.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.gpu_attention import AttentionReducer
from anamnesis.extraction.gpu_families import FamilyReducer
from anamnesis.extraction.gpu_ops import (
    FeatureCollector,
    rowcos,
    std,
    trajectory_indices,
)
from anamnesis.extraction.model_loader import LoadedModel, decoder_layers


@dataclass(frozen=True)
class GpuFeatureResult:
    features: np.ndarray
    feature_names: tuple[str, ...]
    mean_logprob: float
    knnlm_baseline: np.ndarray | None
    metadata: dict


class GpuFeatureLane:
    def __init__(
        self,
        extraction: ExtractionConfig,
        families: FeaturePipelineConfig,
        feature_names: list[str],
        positional_means: np.ndarray,
        pca_components: np.ndarray,
        pca_mean: np.ndarray,
        *,
        device: str | torch.device,
        calibration_sha256: str,
    ):
        required = (
            "include_baseline_tiers",
            "enable_residual_trajectory",
            "enable_attention_flow",
            "enable_gate_features",
            "enable_per_head",
            "enable_value_geometry",
            "enable_qk_geometry",
            "enable_kv_cka",
        )
        if not all(getattr(families, name) for name in required):
            raise ValueError("GPU lane requires the complete declared battery")
        if any(
            getattr(families, name)
            for name in (
                "enable_temporal_dynamics",
                "enable_contrastive_projection",
                "enable_path_signature",
            )
        ):
            raise ValueError("GPU lane does not support additional feature families")
        if not all(
            (
                extraction.enable_tier1,
                extraction.enable_tier2,
                extraction.enable_tier2_5,
                extraction.enable_tier3,
            )
        ):
            raise ValueError("GPU lane cannot drop baseline feature families")
        if len(calibration_sha256) != 64:
            raise ValueError("calibration SHA256 is mandatory")
        self.device = torch.device(device)
        self.config = extraction
        self.families = families
        self.names = tuple(feature_names)
        self.pm = positional_means
        self.pca = torch.as_tensor(
            pca_components, device=self.device, dtype=torch.float64
        )
        self.pca_mean = torch.as_tensor(
            pca_mean, device=self.device, dtype=torch.float64
        )
        self.calibration_sha256 = calibration_sha256
        sources = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("gpu_ops.py"),
                Path(__file__).with_name("gpu_attention.py"),
                Path(__file__).with_name("gpu_families.py"),
            ]
        }
        identity = dict(
            sources=sources,
            extraction=extraction.model_dump(mode="json"),
            families=families.model_dump(mode="json"),
            calibration=calibration_sha256,
            device_type=self.device.type,
            torch=torch.__version__,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            tf32=torch.backends.cuda.matmul.allow_tf32,
        )
        self.identity = identity
        self.lane_id = (
            "torch-eager-reduce-v1-"
            + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
                :20
            ]
        )

    def _outputs(self, out: FeatureCollector, logits: Tensor, chosen: Tensor) -> Tensor:
        entropy, top1, top5, ranks, prob, logprob = [], [], [], [], [], []
        for first in range(0, len(logits), 16):
            x = logits[first : first + 16].float().double()
            x = x - x.max(dim=1, keepdim=True).values
            exp = x.exp()
            z = exp.sum(dim=1)
            logz = z.log()
            cid = chosen[first : first + 16, None]
            xc = x.gather(1, cid).squeeze(1)
            entropy.append((logz - (x * exp).sum(dim=1) / z).float())
            top1.append((exp.max(dim=1).values / z).float())
            top5.append((exp.topk(5, dim=1).values.sum(dim=1) / z).float())
            ranks.append((x > xc[:, None]).sum(dim=1).float())
            prob.append((exp.gather(1, cid).squeeze(1) / z).float())
            logprob.append(xc - logz)
        ent, top, topfive, rank, chosen_prob = [
            torch.cat(v) for v in (entropy, top1, top5, ranks, prob)
        ]
        surprise = -chosen_prob.clamp_min(1e-10).log()
        for name, values in (("logit_entropy", ent), ("top1_prob", top)):
            out.moments(name, values)
            for i, t in enumerate(
                trajectory_indices(len(logits), self.config.trajectory_points)
            ):
                out.put(f"{name}_traj{i}", values[t])
        out.put("top5_mass_mean", topfive.mean())
        for name, values in (("chosen_rank", rank), ("surprise", surprise)):
            out.put("mean_" + name, values.mean())
            out.put("std_" + name, std(values))
        for i, t in enumerate(
            trajectory_indices(len(logits), self.config.trajectory_points)
        ):
            out.put(f"surprise_traj{i}", surprise[t])
        window = self.config.surprise_window
        if len(logits) >= window:
            values = surprise.unfold(0, window, 1)
            running_mean = values.double().mean(dim=1)
            running_std = std(values, dim=1).float()
            threshold = (
                running_mean
                + self.config.surprise_threshold_sigma
                * running_std.double().clamp_min(1e-6)
            )
            out.put(
                "surprise_boundary_count",
                (surprise[window - 1 :].double() > threshold).sum(),
            )
        else:
            out.put("surprise_boundary_count", 0.0)
        return torch.cat(logprob).mean()

    @torch.no_grad()
    def replay_span(
        self, loaded: LoadedModel, token_ids: list[int], start: int, end: int
    ) -> GpuFeatureResult:
        if not 0 < start < end <= len(token_ids) or end - start < 2:
            raise ValueError(
                "invalid span; a nonempty full prefix and >=2 tokens are required"
            )
        if loaded.model.training or loaded.model.config.model_type != "llama":
            raise ValueError("GPU lane requires an eval-mode dense Llama")
        if loaded.model.config._attn_implementation != "eager":
            raise ValueError("GPU lane requires eager attention")
        if any(p.device != self.device for p in loaded.model.parameters()):
            raise ValueError("v1 requires a model entirely on the declared device")
        if end - 2 >= self.pm.shape[1]:
            raise ValueError("positional calibration does not cover span")
        if (
            min(token_ids[:end]) < 0
            or max(token_ids[:end]) >= loaded.model.config.vocab_size
        ):
            raise ValueError("token outside model vocabulary")
        steps = end - start - 1
        collector = FeatureCollector(self.device)
        attention = AttentionReducer(
            collector,
            steps=steps,
            prefix_length=start,
            sampled_layers=self.config.sampled_layers,
            spectral_stride=self.config.spectral_subsample_step,
            n_windows=self.families.temporal_n_windows,
            include_stft=self.families.enable_stft,
        )
        reducer = FamilyReducer(collector, self.config, self.families)
        ids = torch.tensor(
            token_ids[:end], device=self.device, dtype=torch.long
        ).unsqueeze(0)
        loaded.clear_hook_state()
        loaded.disable_hooks()
        pre = loaded.model(ids[:, :start], use_cache=True, return_dict=True)
        cache = pre.past_key_values
        del pre
        handles = []
        h2d_bytes = 0
        try:
            layers = decoder_layers(loaded.model)
            for layer, module in enumerate(layers):

                def hook(_module, _args, result, layer=layer):
                    if (
                        not isinstance(result, tuple)
                        or len(result) < 2
                        or result[1] is None
                    ):
                        raise RuntimeError(
                            "installed eager attention does not expose weights to hook"
                        )
                    attention.consume(layer, result[1])

                handles.append(module.self_attn.register_forward_hook(hook))
            loaded.clear_hook_state()
            loaded.enable_hooks()
            result = loaded.model(
                ids[:, start:end],
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                position_ids=torch.arange(start, end, device=self.device).unsqueeze(0),
                cache_position=torch.arange(start, end, device=self.device),
            )
            if attention.seen != set(range(len(layers))):
                raise RuntimeError("not every eager attention layer was reduced")
            if getattr(result, "attentions", None):
                raise RuntimeError("GPU lane unexpectedly retained attention outputs")
            previous = None
            for layer in range(len(layers)):
                h = result.hidden_states[layer + 1][0, :steps].float()
                pm = np.ascontiguousarray(self.pm[layer + 1, start : start + steps])
                h2d_bytes += pm.nbytes
                corrected = h - torch.as_tensor(pm, device=self.device)
                norms = corrected.norm(dim=-1).float()
                collector.put(f"activation_norm_mean_L{layer}", norms.mean())
                collector.put(f"activation_norm_std_L{layer}", std(norms))
                for i, t in enumerate(
                    trajectory_indices(steps, self.config.trajectory_points)
                ):
                    collector.put(f"activation_norm_traj{i}_L{layer}", norms[t])
                if previous is not None:
                    delta = corrected - previous
                    dn = delta.norm(dim=-1).float()
                    dc = rowcos(delta.double(), previous.double()).float()
                    collector.put(f"delta_norm_mean_L{layer - 1}", dn.mean())
                    collector.put(f"delta_norm_std_L{layer - 1}", std(dn))
                    collector.put(f"delta_cosine_mean_L{layer - 1}", dc.mean())
                previous = corrected
                if layer in self.config.sampled_layers:
                    attention.smoothness(layer, corrected)
                if layer in self.families.trajectory_layers:
                    reducer.residual_trajectory(layer, corrected, h[0].norm())
                if layer in self.config.pca_layers:
                    indices = trajectory_indices(
                        steps, self.config.pca_temporal_samples
                    )
                    projections = (
                        corrected[indices].double() - self.pca_mean
                    ) @ self.pca[: self.config.pca_components].T
                    for ti, projection in enumerate(projections):
                        for ci, value in enumerate(projection):
                            collector.put(f"pca_L{layer}_t{ti}_c{ci}", value)
            logprob = self._outputs(
                collector, result.logits[0, :steps], ids[0, start + 1 : end]
            )
            for layer in self.config.sampled_layers:

                def head(name):
                    values = getattr(loaded.hook_state, name).get(layer)
                    if values is None or len(values) != 1:
                        raise RuntimeError(f"missing/duplicate {name} layer {layer}")
                    return values[0][0, :, :steps, :].permute(1, 0, 2).float()

                keys = head("pre_rope_keys")
                reducer.key(layer, keys)
                attention.head_features(layer, keys)
                reducer.value(layer, head("v_proj_values"))
                reducer.query(layer, head("queries"))
                gate = loaded.hook_state.gate_activations.get(layer)
                if gate is None or len(gate) != 1:
                    raise RuntimeError(f"missing gate layer {layer}")
                reducer.gate(layer, gate[0][0, :steps].float())
            reducer.finish()
            vector = collector.finish(list(self.names)).cpu().numpy()
            if not np.isfinite(vector).all():
                raise RuntimeError("nonfinite GPU feature")
            knnlm = (
                result.hidden_states[-1][0, steps - 1].float().cpu().numpy().copy()
                if self.config.enable_knnlm_baseline
                else None
            )
            ancillary = 8 + (knnlm.nbytes if knnlm is not None else 0)
            receipt = dict(
                lane_id=self.lane_id,
                calibration_sha256=self.calibration_sha256,
                channel="probe-free",
                certified=False,
                batch_size=1,
                attention_layers_reduced=len(attention.seen),
                retained_attention_summary_bytes=attention.retained_bytes,
                gate_tie_exception_d2h_bytes=reducer.gate_tie_exception_d2h_bytes,
                explicit_d2h_bytes=vector.nbytes
                + ancillary
                + reducer.gate_tie_exception_d2h_bytes,
                positional_calibration_h2d_bytes=h2d_bytes,
                input_token_h2d_bytes=ids.numel() * ids.element_size(),
            )
            return GpuFeatureResult(
                vector, self.names, float(logprob.cpu()), knnlm, receipt
            )
        finally:
            for handle in handles:
                handle.remove()
            loaded.clear_hook_state()
