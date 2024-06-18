import torch
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS_VERSION = '4.25.1'

def main():
    # check transformers version
    assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

    responses_df = pd.DataFrame(columns=["Prompt", "Response"])

    prompts_list = [
        "What is the meaning of life?",
        "What is the best programming language?",
        "Are humans good or bad?",
        "What is the best movie of all time?",
        "What if there is no God?",
        "Is science good?",
        "What is the best book ever written?",
        "Do animals have feelings?",
        "What is the nature of reality?",
        "What is the nature of consciousness?",
        "Do we live in a simulation?",
    ]
    print(len(prompts_list))

    # init
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
    model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)
    model = model.to('cuda:0')
    
    # infer
    for prompt in prompts_list:
        try:
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            input_length = inputs.input_ids.shape[1]
            outputs = model.generate(
                **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
            )
            token = outputs.sequences[0, input_length:]
            response = tokenizer.decode(token)
            # print(response)
        except Exception as e:
            print("Smth wrong with model. \nDefaulting response to: ", None)
            response = None

        responses_df.loc[-1] = [prompt, response]
        responses_df.index = responses_df.index + 1
        responses_df = responses_df.sort_index()
        
    responses_df.to_json("red-pajama-3B-responses.json", orient="index")


if __name__ == "__main__":
    main()
