from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from pathlib import Path
import chromadb
from unidecode import unidecode

from transformers import AutoTokenizer
import transformers
import torch
import re

import gradio as gr
from config import default_persist_directory, list_llm


def load_doc(list_file_path: list, chunk_size: int, chunk_overlap: int) -> list:
    # Processing for one document only
    # loader = PyPDFLoader(file_path)
    # pages = loader.load()
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits


def create_db(splits, collection_name: str) -> Chroma:
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
        persist_directory=default_persist_directory,
    )
    return vectordb


def load_db() -> Chroma:
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        persist_directory=default_persist_directory,
        embedding_function=embedding
    )
    return vectordb


def initialize_llmchain(
    llm_model, temperature, max_tokens: int, top_k, vector_db: Chroma, progress=gr.Progress()
) -> ConversationChain:
    progress(0.1, desc="Initializing HF tokenizer...")
    # HuggingFacePipeline uses local model
    # Note: it will download model locally...
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    progress(0.5, desc="Initializing HF pipeline...")
    pipeline = transformers.pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # max_length=1024,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=top_k,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(
        pipeline=pipeline, model_kwargs={"temperature": temperature}
    )

    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )
    # retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    retriever = vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        # combine_docs_chain_kwargs={"prompt": your_prompt})
        return_source_documents=True,
        # return_generated_question=False,
        verbose=False,
    )
    progress(0.9, desc="Done!")
    return qa_chain


# Generate collection name for vector database
#  - Use filepath as input, ensuring unicode text
def create_collection_name(filepath: str):
    # Extract filename without extension
    collection_name = Path(filepath).stem
    # Fix potential issues from naming convention
    ## Remove space
    collection_name = collection_name.replace(" ", "-")
    ## ASCII transliterations of Unicode text
    collection_name = unidecode(collection_name)
    ## Remove special characters
    # collection_name = re.findall("[\dA-Za-z]*", collection_name)[0]
    collection_name = re.sub("[^A-Za-z0-9]+", "-", collection_name)
    ## Limit length to 50 characters
    collection_name = collection_name[:50]
    ## Minimum length of 3 characters
    if len(collection_name) < 3:
        collection_name = collection_name + "xyz"
    ## Enforce start and end as alphanumeric character
    if not collection_name[0].isalnum():
        collection_name = "A" + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + "Z"
    print("Filepath: ", filepath)
    print("Collection name: ", collection_name)
    return collection_name


# Initialize database
def initialize_database(
    list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()
):
    # Create list of documents (when valid)
    list_file_path = [x.name for x in list_file_obj if x is not None]
    # Create collection_name for vector database
    progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(list_file_path[0])
    progress(0.25, desc="Loading document...")
    # Load document and create splits
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    # Create or load vector database
    progress(0.5, desc="Generating vector database...")
    # global vector_db
    vector_db = create_db(doc_splits, collection_name)
    progress(0.9, desc="Done!")
    return vector_db, collection_name, "Complete!"


def initialize_LLM(
    llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()
) -> tuple[ConversationChain, str]:
    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name: ", llm_name)
    qa_chain = initialize_llmchain(
        llm_name, llm_temperature, max_tokens, top_k, vector_db, progress
    )
    return qa_chain, "Complete!"


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history


def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    # print("formatted_chat_history",formatted_chat_history)

    # Generate response using QA chain
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    # print ('chat response: ', response_answer)
    # print('DB source', response_sources)

    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    # return gr.update(value=""), new_history, response_sources[0], response_sources[1]
    return (
        qa_chain,
        gr.update(value=""),
        new_history,
        response_source1,
        response_source1_page,
        response_source2,
        response_source2_page,
        response_source3,
        response_source3_page,
    )


def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    # print(file_path)
    # initialize_database(file_path, progress)
    return list_file_path