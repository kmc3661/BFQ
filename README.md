# Beyond Reconstruction Loss in Post-Training Quantization: Balanced Fitting for Large Vision-Language Models ([Paper](https://arxiv.org/abs/2609.34765))

Balanced Fitting Quantization (BFQ) is a post-training quantization method for large vision-language models. It measures the effect of quantizing each layer/component on calibration loss and uses those measurements to allocate the search budget for channel-wise equalization (CWE).

This repository supports InternVL2, LLaVA-OneVision, and Qwen2-VL under W3A16 and W4A8.

## Installation

From the repository root:

```bash
conda create -n bfq python=3.11
conda activate bfq
pip install -r requirements.txt
bash scripts/setup_thirdparty.sh
pip install -e 3rdparty/LLaVA-NeXT
pip install -e 3rdparty/lmms-eval
pip install -e .
```

The third-party repositories can also be installed from existing local checkouts; see [3rdparty/README.md](3rdparty/README.md).

## Calibration data

Prepare a COCO-caption calibration file in JSON or JSONL format and its image directory. Use the same `image` / `conversations` fields as the [MBQ calibration data](https://github.com/thu-nics/MBQ#apply-model-quantization-in-qmllm-package). The scripts select one ordered 64-sample manifest and use those exact samples for both quantization-effect analysis and CWE search.

Set the model and data paths before running the commands below:

```bash
export MODEL=internvl2
export MODEL_ARGS=pretrained=OpenGVLab/InternVL2-8B
export DATA_PATH=/path/to/calibration.json
export IMAGE_FOLDER=/path/to/images
```

Other supported `MODEL` values are `llava_onevision` and `qwen2_vl`; set `MODEL_ARGS` to the matching pretrained checkpoint.
If `llava_onevision` uses a local snapshot path rather than a Hugging Face model ID, append `,model_name=llava-onevision-qwen2-7b-ov` to `MODEL_ARGS` so that LLaVA loads the Qwen backbone.

## Quantization

The scripts measure quantization effects, infer the allocation rule, construct a per-layer policy, and run CWE calibration. The resulting scale cache is saved under `outputs/bfq/` by default.

```bash
# Weight-only quantization
bash scripts/run_bfq_w3a16_calibrate.sh

# Weight-activation quantization
bash scripts/run_bfq_w4a8_calibrate.sh
```

Set `SCALE_PATH` to change the cache location, `N_SAMPLES` to change the shared sample count, or `CALIB_SEED` to change the subset. The scripts save the selected manifest, its source indices and checksums, the effect summary, and the inferred policy in a companion `_auto_rule/` directory. Both W3A16 and W4A8 use `distort=off`; an existing scale cache causes the calibration wrapper to stop rather than silently reuse it.

For a precomputed policy, `main_quant.py --method bfq --run_process --bfq_policy_override_path /path/to/policy.json` runs the calibration step directly. See `python main_quant.py --help` for model, data, and bit-width options.

## Evaluation

Evaluate a saved cache with the matching precision and model. The wrappers use the paper's `mmmu_val` split by default. The five main-table tasks are `mmmu_val`, `vizwiz_vqa_val`, `scienceqa_img`, `chartqa`, and `ai2d`.

```bash
bash scripts/run_bfq_w3a16_eval.sh
bash scripts/run_bfq_w4a8_eval.sh
```

For the full main table, set `TASKS=mmmu_val,vizwiz_vqa_val,scienceqa_img,chartqa,ai2d` before running either wrapper. Set `SCALE_PATH` if the cache is not at the wrapper's default path, and `OUTPUT_PATH` to choose the evaluation output directory.

## Implementation and tests

- The allocation rule is in [`scripts/bfq_auto_rule.py`](scripts/bfq_auto_rule.py).
- Policy construction and BFQ quantization are in [`qmllm/methods/bfq/`](qmllm/methods/bfq/).
- To check the rule's reference cases without a GPU, run `python -m unittest discover -s tests -v`.

The implementation builds on [MBQ](https://github.com/thu-nics/MBQ).

## Citation

```bibtex
@article{kang2026beyond,
  title={Beyond Reconstruction Loss in Post-Training Quantization: Balanced Fitting for Large Vision-Language Models},
  author={Kang, Minchan and Park, Kyeonghye and Sa, Seungyeon and Cho, Seoyoung and Kim, Daeshik and Cho, Yucheol},
  journal={arXiv preprint arXiv:2609.34765},
  year={2026}
}
```
