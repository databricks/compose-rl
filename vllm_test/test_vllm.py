import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM

def load_weights(self, weights: list[tuple[str, torch.Tensor]]):
    self.model_runner.model.load_weights( # type: ignore
        weights=weights,
    )
    
if __name__ == '__main__':
    prompts = [
        "what is RAY?",
        "what is vLLM?",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    model_name = "facebook/opt-125m"
    print(f'loading model {model_name}...')
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    print('load model done')
    llm = LLM(model=model_name)
    for name, p in model.named_parameters():
        llm.collective_rpc(
            load_weights,
            args=([(name, p)],),
        )
    print('load weights done')
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
