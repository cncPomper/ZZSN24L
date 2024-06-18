from itertools import product
import pandas as pd
from ctransformers import AutoModelForCausalLM


def main():

    models_df = pd.read_csv("llama-2-chat-gguf-variants.csv")

    responses_df = pd.DataFrame(columns=["Model Filename", "Prompt", "Response"])

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

    for model_filename, prompt in product(models_df["Name"], prompts_list):
        try:
            # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
            llm = AutoModelForCausalLM.from_pretrained(
                "TheBloke/Llama-2-7B-GGUF",
                # due to some bug in the library, the model file should not contain "-chat" in the name
                model_file=model_filename.replace("-chat", ""),
                model_type="llama",
                gpu_layers=50,
            )
            response = llm(prompt)
        except Exception as e:
            print(
                "Smth wrong with model: ",
                model_filename,
                "\nDefaulting response to: ",
                None,
            )
            response = None

        responses_df.loc[-1] = [model_filename, prompt, response]
        responses_df.index = responses_df.index + 1
        responses_df = responses_df.sort_index()

    responses_df.to_json("llama-2-chat-gguf-responses.json", orient="index")


if __name__ == "__main__":
    main()
