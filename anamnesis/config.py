"""Configuration for Llama 3.1 8B Instruct extraction.

Key parameters:
  - Model: Llama 3.1 8B Instruct (32 layers, 4096 hidden, 32 query heads)
  - Temperature: 0.6 (model's native)
  - EOS tokens: [128001, 128008, 128009] (includes <|eom_id|>)
  - Sampled layers: [0, 8, 16, 20, 24, 28, 31] (proportional depth sampling)
  - dtype: bfloat16 (model's native)
  - 200 samples: 20 topics × 5 modes × 2 reps
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from anamnesis.modes.run4_modes import (
    RUN4_MODE_INDEX as MODE_INDEX,
    RUN4_MODES as PROCESSING_MODES,
)


# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

# Legacy data root — for accessing 3B experiment data from earlier runs.
# Set ANAMNESIS_LEGACY_DATA to override (e.g., path to old phase_0 outputs).
LEGACY_DATA_ROOT = Path(os.environ.get(
    "ANAMNESIS_LEGACY_DATA",
    str(PROJECT_ROOT.parent / "phase_0"),
))

# Run versioning
RUN_NAME: str = os.environ.get("ANAMNESIS_RUN_NAME", "run_8b_baseline")

OUTPUTS_BASE = PROJECT_ROOT / "outputs"
CALIBRATION_DIR = OUTPUTS_BASE / "calibration" / "llama31_8b"
OUTPUTS_DIR = OUTPUTS_BASE / "runs" / RUN_NAME
SIGNATURES_DIR = OUTPUTS_DIR / "signatures"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PROMPTS_PATH = PROJECT_ROOT / "prompts" / "prompt_sets.json"


# ── Model ──────────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """Configuration for Llama 3.1 8B Instruct."""

    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    torch_dtype: str = "bfloat16"  # model's native dtype
    attn_implementation: str = "eager"  # required — flash/sdpa don't return attn weights
    device_map: str = "auto"

    # Architecture constants (Llama 3.1 8B Instruct)
    num_layers: int = 32
    hidden_dim: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 8  # GQA — same ratio as 3B (4:1)
    head_dim: int = 128
    vocab_size: int = 128256


# ── Generation ─────────────────────────────────────────────────────────────────

class GenerationConfig(BaseModel):
    """Parameters for model.generate()."""

    max_new_tokens: int = 512
    temperature: float = 0.6  # model's native (3B used 0.7)
    top_p: float = 0.9
    do_sample: bool = True

    # Llama 3.1 8B Instruct stop tokens:
    #   128001 = <|end_of_text|>
    #   128008 = <|eom_id|> (end of message, new in 3.1)
    #   128009 = <|eot_id|> (end of turn)
    eos_token_ids: list[int] = Field(
        default=[128001, 128008, 128009],
        description="Token IDs that signal end of generation",
    )

    # These MUST be True for extraction
    output_hidden_states: bool = True
    output_attentions: bool = True
    output_logits: bool = True
    return_dict_in_generate: bool = True


# ── Extraction ─────────────────────────────────────────────────────────────────

class ExtractionConfig(BaseModel):
    """Controls which features to extract and how.

    Sampled layers for 8B (32 layers) follow the same proportional
    sampling as 3B (28 layers): [0%, 25%, 50%, 63%, 75%, 88%, 97%],
    denser at 60-80% depth per the Pochinkov finding.

    3B: [0, 7, 14, 18, 21, 24, 27]  for 28 layers
    8B: [0, 8, 16, 20, 24, 28, 31]  for 32 layers
    """

    sampled_layers: list[int] = Field(
        default=[0, 8, 16, 20, 24, 28, 31],
        description="Layers to extract KV cache and spectral features from",
    )

    # Tier 3 PCA layers — proportional to 3B's [7, 14, 18, 21, 24]
    pca_layers: list[int] = Field(
        default=[8, 16, 20, 24, 28],
        description="Layers for residual stream PCA projection",
    )
    pca_components: int = 50
    pca_temporal_samples: int = 5

    # Temporal sampling for trajectories
    trajectory_points: int = 5

    # Spectral features
    spectral_subsample_step: int = 10

    # KV cache epoch detection
    epoch_window_size: int = 50
    epoch_stride: int = 25

    # Bayesian surprise
    surprise_window: int = 20
    surprise_threshold_sigma: float = 1.5

    # kNN-LM baseline
    knnlm_pca_components: int = 100

    # Enable/disable tiers
    enable_tier1: bool = True
    enable_tier2: bool = True
    enable_tier2_5: bool = True
    enable_tier3: bool = True
    enable_knnlm_baseline: bool = True

    # Cross-layer agreement thresholds (proportional to model depth)
    # These replace the hardcoded l<=7 / l>=21 from the 3B model
    early_layer_cutoff: int = Field(
        default=8,
        description="Layers <= this are 'early' for cross-layer agreement (first quarter)",
    )
    late_layer_cutoff: int = Field(
        default=24,
        description="Layers >= this are 'late' for cross-layer agreement (last quarter)",
    )

    # Raw tensor saving (for GPU-free feature iteration)
    save_raw_tensors: bool = Field(
        default=False,
        description="Save raw per-token tensors alongside feature vectors",
    )
    raw_logits_top_k: int = Field(
        default=50,
        description="Number of top logits to save per timestep (saves 99.96% space)",
    )
    raw_hidden_dtype: str = Field(
        default="float16",
        description="Dtype for saved hidden states and attention (float16 halves disk)",
    )


# ── Feature Pipeline (v2) ──────────────────────────────────────────────────────

class FeaturePipelineConfig(BaseModel):
    """Configuration for pluggable feature families (v2 pipeline).

    Controls which feature families are enabled when computing features
    from saved raw tensors. Each family adds features on top of the
    baseline T1/T2/T2.5/T3 tiers.
    """

    # Whether to include baseline tiers (T1/T2/T2.5/T3 from state_extractor)
    include_baseline_tiers: bool = Field(
        default=True,
        description="Include baseline T1/T2/T2.5/T3 features",
    )

    # Residual stream trajectory features
    enable_residual_trajectory: bool = Field(
        default=False,
        description="Extract trajectory features (velocity, curvature, directness)",
    )
    trajectory_layers: list[int] = Field(
        default=[8, 16, 20, 24, 28],
        description="Layers for trajectory feature computation",
    )

    # Contrastive projection
    enable_contrastive_projection: bool = Field(
        default=False,
        description="Apply trained contrastive projection to hidden states",
    )
    contrastive_model_path: Path | None = Field(
        default=None,
        description="Path to trained contrastive projection model (.pt)",
    )
    contrastive_layers: list[int] = Field(
        default=[8, 16, 20, 24, 28],
        description="Layers for contrastive projection",
    )
    contrastive_temporal_samples: int = Field(
        default=5,
        description="Number of temporal samples for contrastive projection",
    )

    # Attention flow features
    enable_attention_flow: bool = Field(
        default=False,
        description="Extract attention flow features (region decomp, head diversity)",
    )

    # Gate features (SwiGLU)
    enable_gate_features: bool = Field(
        default=False,
        description="Extract SwiGLU gate activation features",
    )
    gate_sparsity_threshold: float = Field(
        default=0.01,
        description="Threshold for gate activation to count as 'active'",
    )

    # Temporal dynamics (windowed T2/T2.5 metrics)
    enable_temporal_dynamics: bool = Field(
        default=False,
        description="Extract windowed temporal decomposition of core T2/T2.5 metrics",
    )

    # Temporal operator settings (shared across families)
    temporal_n_windows: int = Field(
        default=4,
        description="Number of temporal windows for windowed stats",
    )
    enable_stft: bool = Field(
        default=True,
        description="Include STFT spectral features in temporal operators",
    )
    stft_nperseg: int = Field(
        default=64,
        description="STFT window length (reduced for short generations)",
    )


# ── Calibration ────────────────────────────────────────────────────────────────

class CalibrationConfig(BaseModel):
    """Settings for positional decomposition calibration."""

    num_calibration_prompts: int = 50
    calibration_max_tokens: int = 512
    positional_means_path: Path = Field(
        default=CALIBRATION_DIR / "positional_means.npz",
    )
    pca_model_path: Path = Field(
        default=CALIBRATION_DIR / "pca_model.pkl",
    )


# ── Model presets ──────────────────────────────────────────────────────────────

class ModelPreset(BaseModel):
    """Per-model architecture + sampling defaults.

    Single source of truth for everything that varies between Llama 3.1 8B
    and Llama 3.2 3B (architecture, sampled layers, decode params,
    calibration paths). Consumed by `scripts.run_extraction` and
    `extraction.feature_pipeline` so layer presets stay in sync across
    extraction and feature recomputation.
    """

    model_id: str
    torch_dtype: str
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    sampled_layers: list[int]
    pca_layers: list[int]
    trajectory_layers: list[int]
    contrastive_layers: list[int]
    early_layer_cutoff: int
    late_layer_cutoff: int
    temperature: float
    eos_token_ids: list[int]
    calibration_dir: Path


MODEL_PRESETS: dict[str, ModelPreset] = {
    "8b": ModelPreset(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype="bfloat16",
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        sampled_layers=[0, 8, 16, 20, 24, 28, 31],
        pca_layers=[8, 16, 20, 24, 28],
        trajectory_layers=[8, 16, 20, 24, 28],
        contrastive_layers=[8, 16, 20, 24, 28],
        early_layer_cutoff=8,
        late_layer_cutoff=24,
        temperature=0.6,
        eos_token_ids=[128001, 128008, 128009],
        calibration_dir=OUTPUTS_BASE / "calibration" / "llama31_8b",
    ),
    "3b": ModelPreset(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype="float16",
        num_layers=28,
        hidden_dim=3072,
        num_attention_heads=24,
        num_kv_heads=8,
        head_dim=128,
        sampled_layers=[0, 7, 14, 18, 21, 24, 27],
        pca_layers=[7, 14, 18, 21, 24],
        trajectory_layers=[7, 14, 18, 21, 24],
        contrastive_layers=[7, 14, 18, 21, 24],
        early_layer_cutoff=7,
        late_layer_cutoff=21,
        temperature=0.7,
        eos_token_ids=[128001, 128009],
        calibration_dir=LEGACY_DATA_ROOT / "outputs" / "calibration",
    ),
}


# ── Experiment ─────────────────────────────────────────────────────────────────

ProcessingMode = Literal[
    "linear",
    "analogical",
    "socratic",
    "contrastive",
    "dialectical",
]

# Canonical mode prompts and index live in `anamnesis.modes.run4_modes` and are
# re-exported at the top of this module so consumers that already import them
# from `anamnesis.config` keep working.


class GenerationSpec(BaseModel):
    """Full specification for a single generation run.

    mode is str (not ProcessingMode literal) to support extended modes
    (structured, compressed, associative) and prompt-swap modes
    (swap_socratic→linear, etc.) from the unified extraction script.
    """

    generation_id: int
    prompt_set: str
    topic: str
    topic_idx: int
    mode: str
    mode_idx: int
    system_prompt: str
    user_prompt: str
    seed: int
    repetition: int = 0


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    outputs_dir: Path = OUTPUTS_DIR
    signatures_dir: Path = SIGNATURES_DIR
    figures_dir: Path = FIGURES_DIR
    prompts_path: Path = PROMPTS_PATH
    metadata_path: Path = Field(default=OUTPUTS_DIR / "metadata.json")
    results_path: Path = Field(default=OUTPUTS_DIR / "results.json")

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
