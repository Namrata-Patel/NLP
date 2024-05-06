import os
import pandas as pd
import re
import time
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
import joblib

device = 'cuda' if torch.cuda.is_available() else 'cpu'

hg_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
model_name = os.getenv("MODEL_NAME")

tokenizer = AutoTokenizer.from_pretrained(hg_model_name)
model = AutoModel.from_pretrained(hg_model_name)

#Read PDF data
def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text

#Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    docs_chunks =text_splitter.create_documents(docs)
    return docs_chunks

#Function to push data to Pinecone
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    PineconeClient(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index

#*********Functions for dealing with Model related tasks...************

#Read dataset for model creation
def read_data(data):
    df = pd.read_csv(data,delimiter=',', header=None)  
    return df

#Create embeddings instance
def get_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name=os.getenv("MODEL_NAME"))
    return embeddings

#Generating embeddings for our input dataset
def create_embeddings(df,embeddings):
    df[2] = df[0].apply(lambda x: embeddings.embed_query(x))
    return df

#Splitting the data into train & test
def split_train_test__data(df_sample):
    # Split into training and testing sets
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
    list(df_sample[2]), list(df_sample[1]), test_size=0.25, random_state=0)
    print(len(sentences_train))
    return sentences_train, sentences_test, labels_train, labels_test

#Get the accuracy score on test data
def get_score(svm_classifier,sentences_test,labels_test):
    score = svm_classifier.score(sentences_test, labels_test)
    return score

#******************* Methods related to Pinecone ###################


#Function to pull index data from Pinecone...
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    PineconeClient(
        api_key=pinecone_apikey,
        environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

def get_index(pinecone_apikey,pinecone_environment,pincode_index_name):
    # configure client
    pc = PineconeClient(
        api_key=pinecone_apikey,
        environment=pinecone_environment
    )
    # connect to index
    index = pc.Index(pincode_index_name)
    time.sleep(1)
    # view index stats
    index.describe_index_stats()
    return index
#Generating embeddings for our input dataset
def create_embeddings_model(df,embeddings):
    df[2] = df[0].apply(lambda x: embeddings.embed_query(x))
    return df

def create_embeddings():
    #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = SentenceTransformerEmbeddings(model_name=os.getenv("MODEL_NAME"))
    return embeddings

def process_result(result, question):
    text = result['metadata']['text']
    if "phone" in question.lower():
        match = re.search(r"Phone:\s*(\d+)", text)
        return match.group(1) if match else ""
    elif "address" in question.lower():
            match = re.search(r"Address :([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)", text)
            return ", ".join(match.groups()) if match else "Address not found"
    elif "emergency" in question.lower():
            match = re.search(r"Emergency:\s*(\d+)", text)
            return match.group(1) if match else ""
        
    elif "email" in question.lower():
        match = re.search(r"Email:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text)
        return match.group(1) if match else ""
    
    elif "departments" in question.lower():
        match = re.search(r"Departments of hospital:\s*(.*?)(?=\n[A-Z]|\Z)", text, re.DOTALL)
        return match.group(1).strip() if match else ""
    return text

def embed_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def answer_question(index, question,k=1):
    # Embed the question
    question_embedding = embed_text(question)
    
    # Perform the similarity search in the filtered index
    results = index.query(vector=question_embedding, top_k=k, include_metadata=True)
    
    # If no single result dominates, return all relevant results
    for res in results.matches:
         print(res.score)
    return "\n\n".join(process_result(result,question) for result in results.matches)

def predict(query_result):
    Fitmodel = joblib.load('modelsvm.pk1')
    result=Fitmodel.predict([query_result])
    return result[0]