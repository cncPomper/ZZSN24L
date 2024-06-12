import gradio as gr
from gradio_multimodalchatbot import MultiModalChatbot


example = MultiModalChatbot().example_value()

with gr.Blocks() as demo:
    with gr.Row():
        MultiModalChatbot(label="Blank"),  # blank component
        MultiModalChatbot(value=example, label="Populated"),  # populated component


if __name__ == "__main__":
    demo.launch()
