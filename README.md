# Spider_RAG

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser



st.set_page_config(
    page_title="SPIDER ML TASK 2",
    page_icon="🕷️",
    layout="wide"
)

st.markdown("""
Type in a question from the following 4 papers: Attention is all you need, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,GPT-3: Language Models are Few-Shot Learners, LLaMA: Open and Efficient Foundation Language Models""")

os.environ["GROQ_API_KEY"] = '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'

def rag():
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma(
    persist_directory="C:/Users/Ron/Downloads/chroma_db",
    embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
    )
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant.

    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)
    def format_inputs(inputs):
        question = inputs["question"]
        docs = inputs["context"]  # from retriever
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context, "question": question}
    rag_chain = (
        {
            "context": RunnableLambda(lambda d: d["question"]) | retriever,
            "question": RunnableLambda(lambda d: d["question"])
        }

        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

rag_chain, retriever = rag()
if rag_chain and retriever:
    input_question = st.text_input("Type your question here:")
    if input_question:
        answer = rag_chain.invoke({"question": input_question})
        st.subheader("The answer is ")
        st.write(answer)
