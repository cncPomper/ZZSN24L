from langchain_core.prompts import PromptTemplate
import torch
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import fitz  # PyMuPDF for reading PDFs

MIN_TRANSFORMERS_VERSION = '4.25.1'

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text

def main():
    
    assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

    # Extract data from PDF
    pdf_path = r'inputs\simulation.pdf'  # Update with your PDF path
    pdf_data = extract_text_from_pdf(pdf_path)

    # init
    responses_df = pd.DataFrame(columns=["Prompt", "Response"])
    # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    # model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)
    model = model.to('cuda:0')

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
    print("len prompts list", len(prompts_list))

 
    template = PromptTemplate.from_template(
        "You are a helpful AI assistant. Write response to prompt: '{prompt}'. If you don't know, do not make up facts. If needed, use data from {pdf_data}")
    
    # infer
    for prompt in prompts_list:
        try:
            # manual limit for now
            # pdf_data = pdf_data[:1000]  # Limiting the data to 1000 characters

            prepared_prompt = template.format(pdf_data=pdf_data, prompt=prompt)

            inputs = tokenizer(prepared_prompt, return_tensors='pt').to(model.device)
            input_length = inputs.input_ids.shape[1]
            print('input_length:', input_length)
            outputs = model.generate(
                **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
            )
            token = outputs.sequences[0, input_length:]
            response = tokenizer.decode(token)
        except Exception as e:
            print(f"Something went wrong with the model for prompt: {prompt}. Defaulting response to None.")
            response = None

        responses_df.loc[-1] = [prompt, response]
        responses_df.index = responses_df.index + 1
        responses_df = responses_df.sort_index()
        
    responses_df.to_json("llama-2-7b-responses.json", orient="index")


if __name__ == "__main__":
    main()
