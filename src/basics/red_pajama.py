import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS_VERSION = "4.25.1"


def main():
    # check transformers version
    assert (
        transformers.__version__ >= MIN_TRANSFORMERS_VERSION
    ), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."

    # init
    tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16
    )
    model = model.to("cuda:0")
    # infer
    prompt = "<human>: Who is Alan Turing?\n<bot>:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        return_dict_in_generate=True,
    )
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
    print(output_str)


if __name__ == "__main__":
    main()
