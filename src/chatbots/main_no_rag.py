import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from threading import Thread

# tokenizers = [
#     AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1"),
#     AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium"),
#     AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B"),
# ]

# models = [
#     AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16),
#     AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium"),
#     AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B"),
# ]

# also, check other models, like:
# https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
# https://huggingface.co/mistralai/Codestral-22B-v0.1
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# https://huggingface.co/docs/transformers/model_doc/llama3

tokenizer = AutoTokenizer.from_pretrained(
    # "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
    "meta-llama/Llama-2-7b-chat-hf"
)
model = AutoModelForCausalLM.from_pretrained(
    # "togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16
    "meta-llama/Llama-2-7b-chat-hf"
)

model = model.to("cuda:0")

class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(
        [
            "".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
            for item in history_transformer_format
        ]
    )

    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop]),
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != "<":
            partial_message += new_token
            yield partial_message


gr.ChatInterface(predict).launch()
