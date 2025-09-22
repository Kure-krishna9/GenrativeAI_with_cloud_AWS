import json
import boto3
import sys
import os

### Will use Titan Embeding model for genrating embeding

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data engetion
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader


## Vectore embedings  and vectore store

from langchain.vectorstores import FAISS
##LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

#data Ingettion
def data_ingetion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    #-- in our testingcharector split workes better with the PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs=text_splitter.split_documents(documents)
    return docs



## Vector docs and vectore store

def get_vectore_store(docs):
    vectore_store_faiss=FAISS.from_documents(docs,bedrock_embeddings)
    vectore_store_faiss.save_local("faiss_index")
    # return vectore_store_faiss




def get_cloudee_llm():
    ## create the Anthropic model
    llm=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock,
                model_kwargs={'maxTokens':512})
    return llm

def get_llama2_llm():
    ## create the LLAMA2 model
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
                model_kwargs={'max_gen_len':512})
    return llm

prompt_template="""  
  Human:use the following pieces of context to provided a concise answer to the questions at the 
  <context>
  {context}
  <context
  Question:{question}
  Assistant:


"""
Prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

def get_response_llm(llm,vectore_store_faiss,query):
    qa=retrieval_qa.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectore_store_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":Prompt}
    )

    answer=qa({"query":query})
    return answer['result']



import streamlit as st
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF Using AWS Bedrock")

    user_question=st.text_input("Ask a Question From the PDF file")

    with st.sidebar:
        st.title("Update or create vectore store")

        if st.button("Vectore Update"):
            with st.spinner("Processing..."):
                docs=data_ingetion()
                get_vectore_store(docs)
                st.success("Done")

    if st.button("Cloude Output"):
        with st.spinner("Processing...."):
            faiss_index=FAISS.load_local("Faiss_index",bedrock_embeddings)
            llm=get_cloudee_llm()
            # faiss_index=get_vectore_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")
                  
    if st.button("LLama2 Output"):
        with st.spinner("Processing...."):
            faiss_index=FAISS.load_local("Faiss_index",bedrock_embeddings)
            llm=get_llama2_llm()
            # faiss_index=get_vectore_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")


if __name__=='__main__':
    main()
