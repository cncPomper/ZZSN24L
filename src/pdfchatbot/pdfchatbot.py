import yaml

# import fitz
import torch
import json
import gradio as gr
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM
from ctransformers import AutoTokenizer

from sentence_transformers import SentenceTransformer


class PDFChatBot:
    def __init__(self):
        """
        Initialize the PDFChatBot instance.
        """
        self.processed = False
        self.page = 0
        self.chat_history = []
        self.prompt = None
        self.documents = None
        self.embeddings = None
        self.vectordb = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.chain = None
        # self.set_device()

    def add_text(self, history, text):
        """
        Add user-entered text to the chat history.

        Parameters:
            history (list): List of chat history tuples.
            text (str): User-entered text.

        Returns:
            list: Updated chat history.
        """
        print("adding text")
        if not text:
            raise gr.Error("Enter text")
        history.append((text, ""))
        return history

    def create_prompt_template(self):
        """
        Create a prompt template for the chatbot.
        """
        print("creating prompt template")
        template = (
            f"The assistant should provide detailed explanations."
            "Combine the chat history and follow up question into "
            "Follow up question: What is this"
        )
        self.prompt = PromptTemplate.from_template(template)

    def load_embeddings(self):
        """
        Load embeddings from Hugging Face and set in the config file.
        """
        print("loading embeddings")
        # self.embeddings = SentenceTransformer(
        #     'sentence-transformers/all-MiniLM-L6-v2'
        #     )
        self.embeddings = HuggingFaceEmbeddings(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_vectordb(self):
        """
        Load the vector database from the documents and embeddings.
        """
        print("loading vectordb")
        self.vectordb = Chroma.from_documents(
            self.documents, self.embeddings, persist_directory="vectordb"
        )

    def load_tokenizer(self):
        """
        Load the tokenizer from Hugging Face and set in the config file.
        """
        print("loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            # "meta-llama/Llama-2-7b-chat-hf"
            "meta-llama/Llama-2-7b-chat-hf"
        )

    def load_model(self):
        """
        Load the causal language model from Hugging Face and set in the config file.
        """

        self.model = AutoModelForCausalLM.from_pretrained(
            # "meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16
            "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
            torch_dtype=torch.float16,
        )

    def create_pipeline(self):
        """
        Create a pipeline for text generation using the loaded model and tokenizer.
        """
        pipe = pipeline(
            model=self.model,
            task="text-generation",
            tokenizer=self.tokenizer,
            max_new_tokens=200,
        )
        self.pipeline = HuggingFacePipeline(pipeline=pipe)

    def create_chain(self):
        """
        Create a Conversational Retrieval Chain
        """
        self.chain = ConversationalRetrievalChain.from_llm(
            self.pipeline,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 1}),
            condense_question_prompt=self.prompt,
            return_source_documents=True,
        )

    def set_device(self):
        """
        Set the device to 'cuda:0' if available, else set to 'cpu'.
        """
        self.model = (
            self.model.to("cuda:0")
            if torch.cuda.is_available()
            else self.model.to("cpu")
        )

    def process_file(self, file):
        """
        Process the uploaded PDF file and initialize necessary components: Tokenizer, VectorDB and LLM.

        Parameters:
            file (FileStorage): The uploaded PDF file.
        """
        self.create_prompt_template()
        with open("debug.txt", "a") as f:
            f.write("debug - prompt_created")
        self.documents = PyPDFLoader(file.name).load()
        self.load_embeddings()
        with open("debug.txt", "a") as f:
            f.write("debug - embeddings_loaded\n")
        self.load_vectordb()
        with open("debug.txt", "a") as f:
            f.write("debug - vectordb_loaded\n")
        self.load_tokenizer()
        with open("debug.txt", "a") as f:
            f.write("debug - tokenizer_loaded\n")
        self.load_model()
        with open("debug.txt", "a") as f:
            f.write("debug - model_loaded\n")
        self.create_pipeline()
        with open("debug.txt", "a") as f:
            f.write("debug - pipeline_created\n")
        self.create_chain()
        with open("debug.txt", "a") as f:
            f.write("debug - chain_created\n")

    def generate_response(self, history, query, file):
        """
        Generate a response based on user query and chat history.

        Parameters:
            history (list): List of chat history tuples.
            query (str): User's query.
            file (FileStorage): The uploaded PDF file.

        Returns:
            tuple: Updated chat history and a space.
        """
        if not query:
            raise gr.Error(message="Submit a question")
        if not file:
            raise gr.Error(message="Upload a PDF")
        if not self.processed:
            self.process_file(file)
            self.processed = True

        result = self.chain(
            {"question": query, "chat_history": self.chat_history},
            return_only_outputs=True,
        )
        self.chat_history.append((query, result["answer"]))
        self.page = list(result["source_documents"][0])[1][1]["page"]

        for char in result["answer"]:
            history[-1][-1] += char
        return history, " "
