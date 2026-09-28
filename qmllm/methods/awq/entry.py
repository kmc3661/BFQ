import torch

from qmllm.methods.awq.quantize.pre_quant import run_awq, apply_awq
from qmllm.methods.awq.quantize.quantizer import pseudo_quantize_model_weight


def awq_entry(model, prompt_tokens, prompt_mask, run_awq_process: bool, scale_path: str=None, zero_point: str=True, q_group_size: int=128, w_bit: int=4):
    '''
    model: here the model is the LLM, you have to extract the LLM first! 
    prompt_tokens: the prompt tokens
    prompt_mask: the prompt mask, mask the answer language tokens
    run_awq_process: whether to run the AWQ process
    '''
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }

    assert scale_path is not None

    # reparameterization
    if run_awq_process:
        awq_results = run_awq(
            model,
            prompt_tokens,
            prompt_mask,
            w_bit=w_bit,
            q_config=q_config,
            auto_scale=True,
        )
        torch.save(awq_results, scale_path)
        print("AWQ results saved at", scale_path)
    else:
        awq_results = torch.load(scale_path, map_location="cpu")
        apply_awq(model, awq_results)

    # weight quantization
    pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)
        
    return model
