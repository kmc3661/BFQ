# BFQ: Balanced Fitting Quantization for Large Vision-Language Models

This repository contains the open-source release of **BFQ**, a post-training quantization method for large vision-language models that allocates channel-wise equalization budgets using **calibration-set quantization effects**.

**TL;DR.** BFQ measures layer- and component-wise quantization effects on the calibration set, automatically applies a quantization effect-guided allocation rule, and then runs reconstruction-based calibration with adaptive per-layer search budgets.

## Highlights

- Supports `internvl2`, `llava_onevision`, and `qwen2_vl`
- Supports both `W3A16` and `W4A8`
- Public BFQ calibration wrappers automatically apply the quantization effect-guided allocation rule
- Uses the provided **calibration data** for both quantization-effect analysis and scale search
- Supports evaluation through `lmms-eval` with per-sample logging

## Installation

1. Clone the repository.

```bash
git clone https://github.com/kmc3661/BFQ.git
cd BFQ
```

2. Create a Python environment.

```bash
conda create -n bfq python=3.11
conda activate bfq
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
bash scripts/setup_thirdparty.sh
pip install -e 3rdparty/LLaVA-NeXT
pip install -e 3rdparty/lmms-eval
pip install -e .
```

See [3rdparty/README.md](./3rdparty/README.md) if you prefer to point BFQ to external source trees through environment variables.

## Command-Line Interface

### Main quantization entrypoint

`main_quant.py` performs calibration-time quantization search.
When used directly, `main_quant.py` performs BFQ calibration with the arguments you provide.
To reproduce the public BFQ pipeline with automatic quantization-effect analysis and policy generation, use the provided calibration wrappers or pass a pre-built `--glmi_policy_override_path`.

Important arguments:

- `--model`: one of `internvl2`, `llava_onevision`, `qwen2_vl`
- `--model_args`: model loading arguments such as `pretrained=OpenGVLab/InternVL2-8B`
- `--calib_data`: `coco` or `pileval`
- `--data_path`: calibration JSON / JSONL path
- `--image_folder`: image root for multimodal calibration data
- `--n_samples`: number of calibration samples used for reconstruction calibration
- `--method bfq`: enable BFQ
- `--glmi_policy_n_samples`: number of calibration samples used only for quantization-effect analysis (the public BFQ wrappers default to `64`)
- `--scale_path`: path used to save the calibrated quantization results

### Main evaluation entrypoint

`main.py` evaluates the pseudo-quantized model on downstream benchmarks.
Use `--method bfq --pseudo_quant` together with the saved `--scale_path`.

## Run BFQ Calibration

### Recommended: wrapper-based calibration with automatic allocation rule

The public calibration wrappers run calibration-set quantization-effect analysis, infer the allocation rule automatically, build a policy override, and then launch BFQ calibration:

```bash
bash scripts/run_bfq_w3a16_calibrate.sh
bash scripts/run_bfq_w4a8_calibrate.sh
```

They use environment variables such as `MODEL`, `MODEL_ARGS`, `DATA_PATH`, `IMAGE_FOLDER`, and `SCALE_PATH` for customization. By default, they use `64` samples for both quantization-effect analysis and reconstruction calibration.

### Low-level BFQ calibration with a pre-built policy override

If you already have a policy override JSON, you can call `main_quant.py` directly:

```bash
python -W ignore main_quant.py \
  --model internvl2 \
  --model_args pretrained=OpenGVLab/InternVL2-8B \
  --calib_data coco \
  --data_path data/path/calibration.json \
  --image_folder data/path/images \
  --n_samples 64 \
  --method bfq \
  --run_process \
  --w_bit 3 \
  --a_bit 16 \
  --w_group 128 \
  --glmi_policy_override_path outputs/bfq/policy_override.json \
  --scale_path outputs/bfq/internvl2_w3a16.pt
```

## Run Evaluation

### W3A16 evaluation

```bash
python -W ignore main.py \
  --model internvl2 \
  --model_args pretrained=OpenGVLab/InternVL2-8B \
  --tasks mmmu \
  --batch_size 1 \
  --method bfq \
  --pseudo_quant \
  --w_bit 3 \
  --a_bit 16 \
  --w_group 128 \
  --log_samples \
  --log_samples_suffix mmmu \
  --output_path outputs/eval/internvl2_w3a16 \
  --scale_path outputs/bfq/internvl2_w3a16.pt
```

### W4A8 evaluation

```bash
python -W ignore main.py \
  --model internvl2 \
  --model_args pretrained=OpenGVLab/InternVL2-8B \
  --tasks mmmu \
  --batch_size 1 \
  --method bfq \
  --pseudo_quant \
  --w_bit 4 \
  --a_bit 8 \
  --log_samples \
  --log_samples_suffix mmmu \
  --output_path outputs/eval/internvl2_w4a8 \
  --scale_path outputs/bfq/internvl2_w4a8.pt
```

Convenience scripts:

```bash
bash scripts/run_bfq_w3a16_eval.sh
bash scripts/run_bfq_w4a8_eval.sh
```

## Optional: Standalone Quantization-Effect Analysis

The public calibration wrappers apply the rule automatically, so this step is optional.
If you want to inspect the calibration-set quantization effects directly, you can run:

```bash
python scripts/analyze_calib_quantization_effect.py \
  --model internvl2 \
  --model_args pretrained=OpenGVLab/InternVL2-8B \
  --data_json data/path/calibration.json \
  --image_root data/path/images \
  --n_samples 64 \
  --output_dir outputs/quant_effect/internvl2_w4a8 \
  --w_bit 4 \
  --act_a_bit 8
```

This script always analyzes the provided calibration data and writes a `summary.csv` together with per-layer details.

## Notes

- The allocation rule is implemented in `scripts/bfq_auto_rule.py` and is used by default by both calibration wrappers. W3A16 and W4A8 use the corresponding normalization constants and formulas; no model-name lookup is used.
- `GLMI_BUDGET_BONUS_HIGH_PERCENTILE=90` is retained for interface compatibility; the common upper-percentile scale cancels when normalized allocation weights are computed.
- Run the CPU-only rule regression checks with `python -m unittest discover -s tests -v`.

- BFQ-specific optional knobs currently keep the historical `glmi_*` argument names for backward compatibility.
- The public BFQ calibration wrappers analyze the provided calibration data with `64` samples by default, build the quantization effect-guided allocation policy automatically, and then run BFQ calibration.
- If you call `main_quant.py` directly, automatic policy generation is not triggered unless you provide `--glmi_policy_override_path` or manually set the BFQ/GLMI budget arguments.

