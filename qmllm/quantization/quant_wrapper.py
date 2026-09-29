import os

from qmllm.methods.awq.entry import awq_entry
from qmllm.methods.smoothquant.entry import smoothquant_entry
from qmllm.methods.mbq.entry import mbq_entry
from qmllm.methods.qvlm.entry import qvlm_entry
from qmllm.methods.rtn.entry import rtn_entry
from qmllm.methods.qig.entry import qig_entry
from qmllm.methods.qig_official.entry import qig_entry as qig_official_entry
from qmllm.methods.lagq.entry import lagq_entry
from qmllm.methods.bfq.entry import bfq_entry

def qwrapper(model, prompt_inputs, prompt_kwargs, args):
    if args.method == "awq":
        model = awq_entry(model, prompt_inputs, prompt_kwargs, run_awq_process=args.run_process, scale_path=args.scale_path, q_group_size=args.w_group, w_bit=args.w_bit)
    elif args.method == "smoothquant":
        model = smoothquant_entry(model, prompt_inputs, prompt_kwargs, run_sq_process=args.run_process, pseudo_quant=args.pseudo_quant, scale_path=args.scale_path, w_bit=args.w_bit, a_bit=args.a_bit, alpha=args.alpha)
    elif args.method == "mbq":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = mbq_entry(model, prompt_inputs, prompt_kwargs, 
                                run_mbq_process=args.run_process, 
                                pseudo_quant=args.pseudo_quant, 
                                scale_path=args.scale_path, 
                                q_group_size=args.w_group, 
                                w_bit=args.w_bit, 
                                a_bit=args.a_bit, 
                                wa_quant=wa_quant, 
                                reweight=args.reweight,
                                distort=args.distort,
                                loss_mode=args.loss_mode,
                                mbq_debug_grad_samples=getattr(args, "mbq_debug_grad_samples", 0),
                                mbq_debug_max_layers=getattr(args, "mbq_debug_max_layers", 0))
    elif args.method == "qvlm":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = qvlm_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_qvlm_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            wa_quant=wa_quant,
            loss_mode=args.loss_mode,
        )
    elif args.method == "bfq":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = bfq_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_bfq_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            wa_quant=wa_quant,
            distort=args.distort,
            loss_mode=args.loss_mode,
            bfq_policy_n_samples=getattr(args, "bfq_policy_n_samples", None) or args.n_samples,
            bfq_micro_batch_size=getattr(args, "bfq_micro_batch_size", 4),
            bfq_probe_a_bit=getattr(args, "bfq_probe_a_bit", 8),
            bfq_beta=getattr(args, "bfq_beta", 0.5),
            bfq_n_grid_max=getattr(args, "bfq_n_grid_max", 48),
            bfq_n_grid_target_mean=getattr(args, "bfq_n_grid_target_mean", 20.0),
            bfq_budget_conflict_alpha=getattr(args, "bfq_budget_conflict_alpha", 0.5),
            bfq_budget_temperature=getattr(args, "bfq_budget_temperature", 8.0),
            bfq_aggressiveness_temperature=getattr(args, "bfq_aggressiveness_temperature", 3.0),
            bfq_ratio_upper_min=getattr(args, "bfq_ratio_upper_min", 0.5),
            bfq_ratio_upper_max=getattr(args, "bfq_ratio_upper_max", 1.0),
            bfq_modality_ratio_min=getattr(args, "bfq_modality_ratio_min", 0.25),
            bfq_modality_ratio_max=getattr(args, "bfq_modality_ratio_max", 4.0),
            bfq_weight_signed_clip_percentile=getattr(args, "bfq_weight_signed_clip_percentile", 100.0),
            bfq_fixed_budget=getattr(args, "bfq_fixed_budget", False),
            bfq_fixed_aggressiveness=getattr(args, "bfq_fixed_aggressiveness", False),
            bfq_upper_tail_budget=getattr(args, "bfq_upper_tail_budget", False),
            bfq_budget_base_grid=getattr(args, "bfq_budget_base_grid", 16),
            bfq_budget_bonus_low_percentile=getattr(args, "bfq_budget_bonus_low_percentile", 50.0),
            bfq_budget_bonus_high_percentile=getattr(args, "bfq_budget_bonus_high_percentile", 95.0),
            bfq_budget_bonus_gamma=getattr(args, "bfq_budget_bonus_gamma", 2.0),
            bfq_budget_bonus_threshold_mode=getattr(args, "bfq_budget_bonus_threshold_mode", "percentile"),
            bfq_budget_bonus_threshold_alpha=getattr(args, "bfq_budget_bonus_threshold_alpha", 1.0),
            bfq_lower_tail_penalty=getattr(args, "bfq_lower_tail_penalty", False),
            bfq_budget_penalty_low_percentile=getattr(args, "bfq_budget_penalty_low_percentile", 5.0),
            bfq_budget_penalty_high_percentile=getattr(args, "bfq_budget_penalty_high_percentile", 20.0),
            bfq_budget_penalty_gamma=getattr(args, "bfq_budget_penalty_gamma", 2.0),
            bfq_conflict_mode=getattr(args, "bfq_conflict_mode", "shared"),
            bfq_budget_target_mode=getattr(args, "bfq_budget_target_mode", "both"),
            bfq_enable_attention_token_weight=getattr(args, "bfq_enable_attention_token_weight", False),
            bfq_importance_cache_path=getattr(args, "bfq_importance_cache_path", None),
            bfq_disable_importance_cache=getattr(args, "bfq_disable_importance_cache", False),
            bfq_importance_only=getattr(args, "bfq_importance_only", False),
            bfq_policy_override_path=getattr(args, "bfq_policy_override_path", None),
            lagq_disable_logit_sensitivity=getattr(args, "lagq_disable_logit_sensitivity", False),
            lagq_disable_attention_token_weight=getattr(args, "lagq_disable_attention_token_weight", False),
            lagq_micro_batch_size=getattr(args, "lagq_micro_batch_size", 0),
            lagq_stage1_metric=getattr(args, "lagq_stage1_metric", "full_logit"),
            lagq_text_attn_correction=getattr(args, "lagq_text_attn_correction", "exposure_baseline_ratio"),
            lagq_vision_query_source=getattr(args, "lagq_vision_query_source", "both"),
            lagq_query_mix_alpha=getattr(args, "lagq_query_mix_alpha", 0.5),
        )
    elif args.method == "rtn":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = rtn_entry(model, pseudo_quant=args.pseudo_quant, wa_quant=wa_quant, q_group_size=args.w_group, w_bit=args.w_bit, a_bit=args.a_bit)
    elif args.method == "qig":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = qig_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_qig_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            zero_point=True,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            wa_quant=wa_quant,
            loss_mode=args.loss_mode,
            distort=args.distort,
            qig_steps=getattr(args, "qig_steps", 8),
            qig_iqr_factor=getattr(args, "qig_iqr_factor", 1.5),
            qig_eps=getattr(args, "qig_eps", 1e-6),
            qig_disable_iqr=getattr(args, "qig_disable_iqr", False),
            qig_use_abs=(not getattr(args, "qig_no_abs", False)),
            lagq_micro_batch_size=getattr(args, "lagq_micro_batch_size", 0),
        )
    elif args.method == "qig_official":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = qig_official_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_qig_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            zero_point=True,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            wa_quant=wa_quant,
            reweight=args.reweight,
            distort=args.distort,
            loss_mode=args.loss_mode,
        )
    elif args.method == "lagq":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = lagq_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_lagq_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            zero_point=True,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            wa_quant=wa_quant,
            loss_mode=args.loss_mode,
            lagq_layer_groups=getattr(args, "lagq_layer_groups", "0-10,11-25,26-35"),
            lagq_noise_bits=getattr(args, "lagq_noise_bits", 8),
            lagq_noise_strength=getattr(args, "lagq_noise_strength", 1.0),
            lagq_micro_batch_size=getattr(args, "lagq_micro_batch_size", 0),
            lagq_stage1_samples=getattr(args, "lagq_stage1_samples", 0),
            lagq_disable_logit_sensitivity=getattr(args, "lagq_disable_logit_sensitivity", False),
            lagq_disable_attention_token_weight=getattr(args, "lagq_disable_attention_token_weight", False),
            lagq_stage1_metric=getattr(args, "lagq_stage1_metric", "full_logit"),
            lagq_text_attn_correction=getattr(args, "lagq_text_attn_correction", "exposure_baseline_ratio"),
            lagq_vision_query_source=getattr(args, "lagq_vision_query_source", "both"),
            lagq_query_mix_alpha=getattr(args, "lagq_query_mix_alpha", 0.5),
        )
    else:
        raise NotImplementedError

    return model
