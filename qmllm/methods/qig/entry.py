from qmllm.methods.mbq_fg.entry import mbq_entry


def qig_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_qig_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    zero_point: bool = True,
    q_group_size: int = 128,
    w_bit: int = 4,
    a_bit: int = 8,
    wa_quant: bool = True,
    loss_mode: str = "mae",
    distort: bool = False,
    qig_steps: int = 8,
    qig_iqr_factor: float = 1.5,
    qig_eps: float = 1e-6,
    qig_disable_iqr: bool = False,
    qig_use_abs: bool = True,
    lagq_micro_batch_size: int = 0,
):
    # QIG follows MBQ's scale-search framework but replaces modality reweighting
    # with token-wise Quantization-aware Integrated Gradients.
    return mbq_entry(
        model=model,
        prompt_inputs=prompt_inputs,
        prompt_kwargs=prompt_kwargs,
        run_mbq_process=run_qig_process,
        pseudo_quant=pseudo_quant,
        scale_path=scale_path,
        zero_point=zero_point,
        q_group_size=q_group_size,
        w_bit=w_bit,
        a_bit=a_bit,
        wa_quant=wa_quant,
        reweight=False,
        distort=distort,
        loss_mode=loss_mode,
        finegrained=True,
        qig_steps=qig_steps,
        qig_iqr_factor=qig_iqr_factor,
        qig_eps=qig_eps,
        qig_disable_iqr=qig_disable_iqr,
        qig_use_abs=qig_use_abs,
        lagq_micro_batch_size=lagq_micro_batch_size,
    )
