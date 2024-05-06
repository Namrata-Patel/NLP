#Pinecone team has been making a lot of changes to there code and here is how it should be used going forward :)
from pinecone import Pinecone as PineconeClient
#from langchain.vectorstores import Pinecone     #This import has been replaced by the below one :)
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain.llms import OpenAI #This import has been replaced by the below one :)
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
#from langchain.callbacks import get_openai_callback #This import has been replaced by the below one :)
from langchain_community.callbacks import get_openai_callback
import joblib
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, AutoModel
from sentence_transformers import SentenceTransformer
import torch

import time
import re

lang_model_name = "./checkpoints/roberta-base-squad2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
#model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


#Function to pull index data from Pinecone...
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    PineconeClient(
        api_key=pinecone_apikey,
        environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

def get_index(pinecone_apikey,pinecone_environment,pincode_index_name,embeddings):
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

def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

########################### GET  SIMILARITY VECTORS FROM STORE ##################
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
    #print(results)
    
    # # Check for proper response handling
    # if not results.matches:
    #     return "No results found."

    # # Process the results based on score and filter irrelevant ones
    # relevant_results = [result for result in results.matches if result.score > 1]

    # # Check the number of relevant results
    # if not relevant_results:
    #     return "No sufficiently relevant results found.", results

    # if relevant_results[0].score > 2.2:
    #     # Return the first two results
    #     return "\n".join(process_result(result, question) for result in relevant_results[:1]) , results

    # If no single result dominates, return all relevant results
    for res in results.matches:
         print(res.score)
    return "\n\n===================\n".join(process_result(result,question) for result in results.matches)

########################### GET  SIMILARITY VECTORS FROM STORE ##################

#This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
def get_similar_docs(index,query,k=4):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_summary(pine_docs,user_input):
    nlp = pipeline('question-answering', model=lang_model_name, tokenizer=lang_model_name)
    docs = [extract_content_from_document(doc) for doc in pine_docs]
    context = " ".join(docs)
    answers = []
    for ctx in docs:
        result = nlp(question=user_input, context=ctx)
        answers.append(result["answer"])

    print(answers)
    return answers
def get_answer1(pine_docs,user_input):
    nlp = pipeline('question-answering', model=lang_model_name, tokenizer=lang_model_name)
    docs = [extract_content_from_document(doc) for doc in pine_docs]
    context = " ".join(docs)
    answers = []
    for ctx in docs:
        result = nlp(question=user_input, context=ctx)
        answers.append(result["answer"])

    print(answers)
    return answers

def get_answer(pine_docs,user_input):
    #nlp = pipeline('question-answering', model=lang_model_name, tokenizer=lang_model_name)
    nlp = pipeline('question-answering', model=lang_model_name, tokenizer=lang_model_name)
    docs = [extract_content_from_document(doc) for doc in pine_docs]

    QA_input = {
    'question': user_input,
    'context': docs
    }

    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(lang_model_name)
    tokenizer = AutoTokenizer.from_pretrained(lang_model_name)

    # Initialize an empty array to store answers
    answers = []

    # Iterate over each context string and find the answer
    for context in QA_input["context"]:
        # Tokenize input question and context
        inputs = tokenizer(QA_input["question"], context, return_tensors="pt")

        # Get start and end logits
        start_logits, end_logits = model(**inputs).start_logits, model(**inputs).end_logits

        # Decode the answer span
        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

        # Skip if answer contains '<s>'
        if '<s>' in answer:
            continue

        # Clean the answer by removing leading or trailing special characters
        answer = answer.strip("<s>").strip("</s>").strip()

        # Add answer to the array
        answers.append(answer)

    if len(answers)<=0:
        print("No answers")


    return answers

# def get_answer(docs,user_input):
#     chain = load_qa_chain(OpenAI(), chain_type="stuff")
#     with get_openai_callback() as cb:
#         response = chain.run(input_documents=docs, question=user_input)
#     return response

def extract_content_from_document(document):
    # Extract content from the page_content attribute
    content = document.page_content
    return content

# def get_answer(pine_docs, user_input):

#     # Convert Pine Docs to String[]
#     docs = [extract_content_from_document(doc) for doc in pine_docs]

#     # Load the QA pipeline with a pre-trained model
#     qa_pipeline = pipeline("question-answering")

#     # Concatenate all documents into one string for easier processing
#     concatenated_docs = " ".join(docs)

#     # Use the QA pipeline to get answers to the user's question
#     answers = qa_pipeline(question=user_input, context=concatenated_docs, top_k=5)

#     # Extract just the 'answer' text from each answer
#     extracted_answers = [answer['answer'] for answer in answers]

#     # Find the start and end indices of each answer in the concatenated_docs
#     answer_indices = [(answer['start'], answer['end']) for answer in answers]

#     # Extract the full sentences containing each answer
#     full_sentences = [get_full_sentence(concatenated_docs, start_idx, end_idx) 
#                       for start_idx, end_idx in answer_indices]

#     # Remove duplicate answers
#     unique_answers = list(set(full_sentences))

#     # Format the unique answers as a list
#     formatted_answers = "\n".join(f"{i+1}. {answer}" for i, answer in enumerate(unique_answers))

#     return formatted_answers

# def get_full_sentence(concatenated_docs, start_idx, end_idx):
#     # Find the start of the sentence containing the answer
#     sentence_start = concatenated_docs.rfind('.', 0, start_idx) + 1

#     # Find the end of the sentence containing the answer
#     sentence_end = concatenated_docs.find('.', end_idx)

#     # Extract the full sentence
#     full_sentence = concatenated_docs[sentence_start:sentence_end].strip()

#     return full_sentence

def predict(query_result):
    Fitmodel = joblib.load('modelsvm.pk1')
    result=Fitmodel.predict([query_result])
    return result[0]