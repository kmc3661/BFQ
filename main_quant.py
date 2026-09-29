import argparse
import datetime
import importlib
import importlib.util
import json
import os
import sys
import traceback
import warnings
from functools import partial

import numpy as np
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from typing import Union


def _prepend_first_existing(paths):
    for path in paths:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
            return path
    return None


def _inject_local_3rdparty_paths():
    try:
        if importlib.util.find_spec("llava") is None:
            _prepend_first_existing(
                [
                    os.environ.get("LLAVA_SRC_PATH", ""),
                    os.path.join(os.path.dirname(__file__), "3rdparty", "LLaVA-NeXT"),
                ]
            )
    except Exception:
        pass

    try:
        lmms_spec = importlib.util.find_spec("lmms_eval")
    except Exception:
        lmms_spec = None
    if lmms_spec is None or "site-packages" in str(getattr(lmms_spec, "origin", "")):
        _prepend_first_existing(
            [
                os.environ.get("LMMS_EVAL_SRC_PATH", ""),
                os.path.join(os.path.dirname(__file__), "3rdparty", "lmms-eval"),
            ]
        )


_inject_local_3rdparty_paths()

from lmms_eval.models import get_model

from qmllm.quantization.quant_wrapper import qwrapper
from qmllm.models import get_process_model
from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset


def parse_quant_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    # calibration parameters
    parser.add_argument("--calib_data", default="pileval", choices=["pileval", "coco", None])
    parser.add_argument("--n_samples", default=128, type=int)
    parser.add_argument("--calib_no_shuffle", action="store_true", help="Read a preselected BFQ calibration manifest in its saved order")
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)

    # TODO: quantization parameters
    parser.add_argument("--method", default="awq", choices=["awq", "smoothquant", "mbq", "qig", "qig_official", "lagq", "rtn", "qvlm", "bfq", None])
    parser.add_argument("--w_bit", default=8, type=int)
    parser.add_argument("--a_bit", default=16, type=int)
    parser.add_argument("--w_group", default=128, type=int)
    parser.add_argument("--alpha", default=0.5, type=int)
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--distort", action="store_true")
    parser.add_argument("--loss_mode", default="mae", choices=["mae", "mse"])
    parser.add_argument(
        "--mbq_debug_grad_samples",
        default=0,
        type=int,
        help="Debug only: limit number of samples used in MBQ gradient/reweight stage (0: use all).",
    )
    parser.add_argument(
        "--mbq_debug_max_layers",
        default=0,
        type=int,
        help="Debug only: run MBQ on first N decoder layers only (0: all layers).",
    )
    parser.add_argument("--qig_steps", default=8, type=int, help="Integrated-gradient steps for QIG method")
    parser.add_argument("--qig_iqr_factor", default=1.5, type=float, help="IQR clipping factor for QIG token scores")
    parser.add_argument("--qig_eps", default=1e-6, type=float, help="epsilon for QIG token weight normalization")
    parser.add_argument("--qig_disable_iqr", action="store_true", help="Disable IQR clipping on QIG scores")
    parser.add_argument("--qig_no_abs", action="store_true", help="Do not take absolute value before QIG token reduction")
    parser.add_argument(
        "--lagq_layer_groups",
        default="0-10,11-25,26-35",
        type=str,
        help="Layer groups for stage-1 sensitivity, e.g. '0-10,11-25,26-35'",
    )
    parser.add_argument("--lagq_noise_bits", default=8, type=int, help="Quant bits for stage-1 sensitivity perturbation")
    parser.add_argument("--lagq_noise_strength", default=1.0, type=float, help="Strength of stage-1 perturbation")
    parser.add_argument("--lagq_micro_batch_size", default=0, type=int, help="Micro-batch size for LAGQ calibration forward/capture (0: full batch)")
    parser.add_argument(
        "--lagq_stage1_samples",
        default=0,
        type=int,
        help="Number of samples used only for LAGQ stage-1 sensitivity (0: use all calibration samples)",
    )
    parser.add_argument("--lagq_disable_logit_sensitivity", action="store_true", help="Disable stage-1 logit sensitivity weighting")
    parser.add_argument(
        "--lagq_disable_attention_token_weight",
        action="store_true",
        help="Disable attention-based token weighting in LAGQ (use only stage-1 layer/modality sensitivity)",
    )
    parser.add_argument(
        "--lagq_stage1_metric",
        default="full_logit",
        choices=["full_logit", "answer_only"],
        help="Stage-1 sensitivity metric: full sequence logits or answer-only supervised tokens",
    )
    parser.add_argument(
        "--lagq_text_attn_correction",
        default="exposure_baseline_ratio",
        choices=["none", "exposure", "baseline_ratio", "baseline_diff", "exposure_baseline_ratio", "exposure_baseline_diff"],
        help="Text received-attention correction mode for LAGQ token weighting",
    )
    parser.add_argument(
        "--lagq_vision_query_source",
        default="both",
        choices=["text", "both"],
        help="Vision token scoring query source for LAGQ",
    )
    parser.add_argument("--lagq_query_mix_alpha", default=0.5, type=float, help="alpha for query_source=both mix")
    parser.add_argument("--bfq_policy_n_samples", default=None, type=int, help="Number of BFQ policy-analysis samples; must match --n_samples when generating a policy directly")
    parser.add_argument("--bfq_micro_batch_size", default=4, type=int, help="Micro-batch size for BFQ quantization-effect policy analysis")
    parser.add_argument("--bfq_probe_a_bit", default=8, type=int, help="Activation bit used only for BFQ vision/text probe analysis")
    parser.add_argument("--bfq_beta", default=0.5, type=float, help="Beneficial-score discount factor in BFQ risk score")
    parser.add_argument("--bfq_n_grid_max", default=48, type=int, help="Maximum per-layer search grid for BFQ")
    parser.add_argument("--bfq_n_grid_target_mean", default=20.0, type=float, help="Target average per-layer search grid for BFQ budget allocation")
    parser.add_argument("--bfq_budget_conflict_alpha", default=0.5, type=float, help="Conflict-score weight in BFQ budget allocation")
    parser.add_argument("--bfq_budget_temperature", default=8.0, type=float, help="Softmax temperature for BFQ budget allocation")
    parser.add_argument("--bfq_aggressiveness_temperature", default=3.0, type=float, help="Temperature for mapping signed BFQ scores into aggressiveness")
    parser.add_argument("--bfq_ratio_upper_min", default=0.5, type=float, help="Conservative upper bound for BFQ scale-ratio search")
    parser.add_argument("--bfq_ratio_upper_max", default=1.0, type=float, help="Aggressive upper bound for BFQ scale-ratio search")
    parser.add_argument("--bfq_modality_ratio_min", default=0.25, type=float, help="Minimum vision/text reweight ratio in BFQ")
    parser.add_argument("--bfq_modality_ratio_max", default=4.0, type=float, help="Maximum vision/text reweight ratio in BFQ")
    parser.add_argument("--bfq_weight_signed_clip_percentile", default=100.0, type=float, help="Percentile for clipping weight signed scores before BFQ budget/aggressiveness mapping")
    parser.add_argument("--bfq_fixed_budget", action="store_true", help="Disable BFQ budget redistribution and fix all per-layer budgets to the target mean")
    parser.add_argument("--bfq_fixed_aggressiveness", action="store_true", help="Disable BFQ aggressiveness adaptation and use a shared ratio upper bound for all layers")
    parser.add_argument("--bfq_upper_tail_budget", action="store_true", help="Use baseline-plus-bonus upper-tail budget allocation instead of softmax redistribution")
    parser.add_argument("--bfq_budget_base_grid", default=16, type=int, help="Baseline per-layer grid before upper-tail bonus allocation")
    parser.add_argument("--bfq_budget_bonus_low_percentile", default=50.0, type=float, help="Lower percentile that starts receiving upper-tail budget bonus")
    parser.add_argument("--bfq_budget_bonus_high_percentile", default=95.0, type=float, help="Upper percentile used to normalize upper-tail budget bonus")
    parser.add_argument("--bfq_budget_bonus_gamma", default=2.0, type=float, help="Power used to emphasize the upper tail in budget bonus allocation")
    parser.add_argument("--bfq_budget_bonus_threshold_mode", default="percentile", choices=["percentile", "auto"], help="How BFQ determines which upper-tail layers receive bonus budget")
    parser.add_argument("--bfq_budget_bonus_threshold_alpha", default=1.0, type=float, help="Distribution-aware threshold strength for BFQ upper-tail auto bonus mode")
    parser.add_argument("--bfq_lower_tail_penalty", action="store_true", help="Allow clearly low-importance layers to drop below the baseline grid before reallocating budget upward")
    parser.add_argument("--bfq_budget_penalty_low_percentile", default=5.0, type=float, help="Lower percentile anchor for lower-tail budget penalty")
    parser.add_argument("--bfq_budget_penalty_high_percentile", default=20.0, type=float, help="Upper percentile anchor for lower-tail budget penalty")
    parser.add_argument("--bfq_budget_penalty_gamma", default=2.0, type=float, help="Power used to emphasize the lower-tail penalty allocation")
    parser.add_argument("--bfq_conflict_mode", default="shared", choices=["shared", "split_vt", "none"], help="How BFQ conflict bonus is applied to weight/activation budgets")
    parser.add_argument("--bfq_budget_target_mode", default="both", choices=["both", "weight_only", "act_only"], help="Which branch receives adaptive budget allocation in W4A8")
    parser.add_argument("--bfq_enable_attention_token_weight", action="store_true", help="Use attention-based token weighting inside the BFQ quantization/search step")
    parser.add_argument("--bfq_importance_cache_path", default=None, type=str, help="Optional opt-in path to a cached BFQ quantization-effect importance file. When not provided, BFQ keeps the original behavior and recomputes probing.")
    parser.add_argument("--bfq_disable_importance_cache", action="store_true", help="Disable BFQ importance cache load/save even when --bfq_importance_cache_path is provided")
    parser.add_argument("--bfq_importance_only", action="store_true", help="Run BFQ quantization-effect policy analysis and cache save only, then exit before MBQ calibration/search")
    parser.add_argument("--bfq_policy_override_path", default=None, type=str, help="Optional path to a prebuilt BFQ policy JSON. When provided, skip policy search and directly use this policy for calibration/search.")
    parser.add_argument("--scale_path", default=None, type=str)
    parser.add_argument("--run_process", action="store_true")
    parser.add_argument("--pseudo_quant", action="store_true")
    args = parser.parse_args()
    return args


def cli_quant(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_quant_args()

    args_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    for args in args_list:
        cli_quant_single(args)


def cli_quant_single(args: Union[argparse.Namespace, None] = None) -> None:
    # here we load MLLMs outside of the evaluator.
    if args.method == "bfq" and args.run_process and not getattr(args, "bfq_policy_override_path", None):
        policy_samples = getattr(args, "bfq_policy_n_samples", None)
        if policy_samples is not None and policy_samples != args.n_samples:
            raise ValueError("BFQ policy analysis and CWE calibration must use the same number of samples")
        args.bfq_policy_n_samples = args.n_samples

    if args.model_args is None:
        args.model_args = ""
    
    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )

    # Preprocess the MLLM here, use "lm._model" to get the fp16 mllm.
    Process_ModelClass = get_process_model(args.model)
    process_model = Process_ModelClass(lm._model, 
                                       lm._tokenizer, 
                                       lm.processor if hasattr(lm, 'processor') else None)

    # Generate the calibration tokens.
    prompt_inputs = None
    prompt_kwargs = None

    if args.calib_data == "pileval":
        prompt_inputs, prompt_kwargs = get_calib_dataset(data_path=args.data_path, tokenizer=lm._tokenizer, n_samples=args.n_samples)
    elif args.calib_data == "coco":
        ordered_manifest = bool(getattr(args, "calib_no_shuffle", False))
        prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(data_path=args.data_path,
                                                                    image_folder=args.image_folder,
                                                                    model=process_model,
                                                                    n_samples=args.n_samples,
                                                                    few_shot_format=args.few_shot_format,
                                                                    interleave_format=args.interleave_format,
                                                                    text_data_path=args.text_data_path,
                                                                    shuffle=not ordered_manifest,
                                                                    require_exact_n_samples=ordered_manifest)

    # Wrapper the quantized model.
    qwrapper(process_model, prompt_inputs, prompt_kwargs, args)

    
if __name__ == "__main__":
    cli_quant()
