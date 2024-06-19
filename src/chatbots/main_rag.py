import gradio as gr
import os
from utils import initialize_database, initialize_LLM, conversation
from config import list_llm

list_llm_simple = [os.path.basename(llm) for llm in list_llm]


def demo():
    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()

        gr.Markdown(
            """<center><h2>RAG demonstration chatbot</center></h2>
        <h3>Ask any questions about your PDF documents</h3>"""
        )
        gr.Markdown(
            """<b>Note:</b> This AI assistant, using Langchain and open-source LLMs, performs retrieval-augmented generation (RAG) from your PDF documents. \
        The user interface explicitely shows multiple steps to help understand the RAG workflow. 
        This chatbot takes past questions into account when generating answers (via conversational memory), and includes document references for clarity purposes.<br>
        """
        )

        with gr.Tab("Step 1 - Upload PDF"):
            with gr.Row():
                document = gr.Files(
                    height=100,
                    file_count="multiple",
                    file_types=["pdf"],
                    interactive=True,
                    label="Upload your PDF documents (single or multiple)",
                )

        with gr.Tab("Step 2 - Process document"):
            with gr.Row():
                db_btn = gr.Radio(
                    ["ChromaDB"],
                    label="Vector database type",
                    value="ChromaDB",
                    type="index",
                    info="Choose your vector database",
                )
            with gr.Accordion("Advanced options - Document text splitter", open=False):
                with gr.Row():
                    slider_chunk_size = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=600,
                        step=20,
                        label="Chunk size",
                        info="Chunk size",
                        interactive=True,
                    )
                with gr.Row():
                    slider_chunk_overlap = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=40,
                        step=10,
                        label="Chunk overlap",
                        info="Chunk overlap",
                        interactive=True,
                    )
            with gr.Row():
                db_progress = gr.Textbox(
                    label="Vector database initialization", value="None"
                )
            with gr.Row():
                db_btn = gr.Button("Generate vector database")

        with gr.Tab("Step 3 - Initialize QA chain"):
            with gr.Row():
                llm_btn = gr.Radio(
                    list_llm_simple,
                    label="LLM models",
                    value=list_llm_simple[0],
                    type="index",
                    info="Choose your LLM model",
                )
            with gr.Accordion("Advanced options - LLM model", open=False):
                with gr.Row():
                    slider_temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Model temperature",
                        interactive=True,
                    )
                with gr.Row():
                    slider_maxtokens = gr.Slider(
                        minimum=224,
                        maximum=4096,
                        value=1024,
                        step=32,
                        label="Max Tokens",
                        info="Model max tokens",
                        interactive=True,
                    )
                with gr.Row():
                    slider_topk = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="top-k samples",
                        info="Model top-k samples",
                        interactive=True,
                    )
            with gr.Row():
                llm_progress = gr.Textbox(value="None", label="QA chain initialization")
            with gr.Row():
                qachain_btn = gr.Button("Initialize Question Answering chain")

        with gr.Tab("Step 4 - Chatbot"):
            chatbot = gr.Chatbot(height=300)
            with gr.Accordion("Advanced - Document references", open=False):
                with gr.Row():
                    doc_source1 = gr.Textbox(
                        label="Reference 1", lines=2, container=True, scale=20
                    )
                    source1_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source2 = gr.Textbox(
                        label="Reference 2", lines=2, container=True, scale=20
                    )
                    source2_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source3 = gr.Textbox(
                        label="Reference 3", lines=2, container=True, scale=20
                    )
                    source3_page = gr.Number(label="Page", scale=1)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type message (e.g. 'What is this document about?')",
                    container=True,
                )
            with gr.Row():
                submit_btn = gr.Button("Submit message")
                clear_btn = gr.ClearButton([msg, chatbot], value="Clear conversation")

        # Preprocessing events
        db_btn.click(
            initialize_database,
            inputs=[document, slider_chunk_size, slider_chunk_overlap],
            outputs=[vector_db, collection_name, db_progress],
        )
        qachain_btn.click(
            initialize_LLM,
            inputs=[
                llm_btn,
                slider_temperature,
                slider_maxtokens,
                slider_topk,
                vector_db,
            ],
            outputs=[qa_chain, llm_progress],
        ).then(
            lambda: [None, "", 0, "", 0, "", 0],
            inputs=None,
            outputs=[
                chatbot,
                doc_source1,
                source1_page,
                doc_source2,
                source2_page,
                doc_source3,
                source3_page,
            ],
            queue=False,
        )

        # Chatbot events
        msg.submit(
            conversation,
            inputs=[qa_chain, msg, chatbot],
            outputs=[
                qa_chain,
                msg,
                chatbot,
                doc_source1,
                source1_page,
                doc_source2,
                source2_page,
                doc_source3,
                source3_page,
            ],
            queue=False,
        )
        submit_btn.click(
            conversation,
            inputs=[qa_chain, msg, chatbot],
            outputs=[
                qa_chain,
                msg,
                chatbot,
                doc_source1,
                source1_page,
                doc_source2,
                source2_page,
                doc_source3,
                source3_page,
            ],
            queue=False,
        )
        clear_btn.click(
            lambda: [None, "", 0, "", 0, "", 0],
            inputs=None,
            outputs=[
                chatbot,
                doc_source1,
                source1_page,
                doc_source2,
                source2_page,
                doc_source3,
                source3_page,
            ],
            queue=False,
        )
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    demo()
