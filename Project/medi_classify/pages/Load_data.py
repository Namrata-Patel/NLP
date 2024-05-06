import os
import streamlit as st
from dotenv import load_dotenv
from utils.utils import *

def main():
    load_dotenv()

    # Display the logo
    # Display the logo with custom styling
    logo_html = f"""
        <div style="display:flex; justify-content:center; align-items:center;">
            <img src='https://velocityhospital.com/wp-content/uploads/2019/08/velocity-1.png' 
            style="width: 200px; height: 200px; object-fit: contain;">
            <h1 style="margin-left: 20px;">Medi-Classify</h1>
        </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)

    # st.set_page_config(page_title="Vector Store Data Loading")
    
    st.title("Please upload pdf files to load into Vector Store...📁 ")

    # st.write(f"PineCone Index=>", os.getenv("PINECONE_INDEX_NAME"))

    # Let the user upload multiple PDF files
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type='pdf')

    if st.button("Process Files"):
        if not uploaded_files:
            st.error("Please upload at least one PDF file.")
            return

        # Process each uploaded PDF file
        for uploaded_file in uploaded_files:
            with st.spinner(f'Processing {uploaded_file.name}...'):
                # Read and process the file
                text = read_pdf_data(uploaded_file)
                st.write(f"👉Reading PDF {uploaded_file.name} done")

                # Create chunks
                docs_chunks = split_data(text)
                st.write(f"👉Splitting data into chunks for {uploaded_file.name} done")

                # Create the embeddings
                embeddings = create_embeddings()
                st.write(f"👉Creating embeddings instance for {uploaded_file.name} done")

                # Build the vector store
                # Assumes "PINECONE_API_KEY" environment variable for Pinecone usage
                

                push_to_pinecone(os.getenv("PINECONE_API_KEY"), "gcp-starter", os.getenv("PINECONE_INDEX_NAME"), embeddings, docs_chunks)

                st.success(f"Successfully pushed the embeddings for {uploaded_file.name} to Pinecone")

if __name__ == '__main__':
    main()
