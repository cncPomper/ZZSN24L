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
    # check transformers version
    assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

    responses_df = pd.DataFrame(columns=["Prompt", "Response"])

    # Extract prompts from PDF
    pdf_path = r'inputs\nike-report.pdf'  # Update with your PDF path
    prompts_text = extract_text_from_pdf(pdf_path)
    prompts_list = prompts_text.split('\n\n')  # Assuming prompts are separated by double newlines

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
        except Exception as e:
            print(f"Something went wrong with the model for prompt: {prompt}. Defaulting response to None.")
            response = None

        responses_df.loc[-1] = [prompt, response]
        responses_df.index = responses_df.index + 1
        responses_df = responses_df.sort_index()
        
    responses_df.to_json("red-pajama-3B-responses.json", orient="index")


if __name__ == "__main__":
    main()
