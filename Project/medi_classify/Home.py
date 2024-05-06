from dotenv import load_dotenv
import streamlit as st
from utils.utils import *

import datetime

#Creating session variables
if 'tickets' not in st.session_state:
    st.session_state['tickets'] =[]




def main():
    load_dotenv()

    import os

    logo_html = f"""
        <div style="display:flex; justify-content:center; align-items:center;">
            <img src='https://velocityhospital.com/wp-content/uploads/2019/08/velocity-1.png' 
            style="width: 200px; height: 200px; object-fit: contain;">
            <h1 style="margin-left: 20px;">Medi-Classify</h1>
        </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)
    
    #Capture user input
    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("🔍")
    

    if user_input:
        
        medi_index = get_index(os.getenv("PINECONE_API_KEY"),"gcp-starter",os.getenv("PINECONE_INDEX_NAME"))

        # Fetch the answer from the vector store
        top_results = answer_question(medi_index,user_input)
        
        st.markdown(f"<b style='color:blue'>Question : </b>{user_input}<br/>",unsafe_allow_html=True)
        st.markdown("<b style='color:blue'>Answer : </b><br/>",unsafe_allow_html=True)

        top_results = top_results.replace("\n","<br/>")
        st.markdown(top_results,unsafe_allow_html=True)
        print(top_results)


        #Button to create a ticket with respective department
        button = st.button("Submit ticket?")

        if button:
            #Get Response
            

            embeddings = create_embeddings()
            query_result = embeddings.embed_query(user_input)

            # Retrieve existing ticket data from session state
            ticket_dict = st.session_state.get('tickets', {})

            department_value = predict(query_result)
            st.write("Your ticket has been submitted to: " + department_value)

            # Get the current time
            ticket_time = datetime.datetime.now()

            # Retrieve existing ticket data from session state
            ticket_dict = st.session_state['tickets']

            if not isinstance(ticket_dict, dict):  # Ensure ticket_dict is a dictionary
                ticket_dict = {}

            # Append ticket to the corresponding department
            if department_value in ticket_dict:
                ticket_dict[department_value].append((ticket_time, user_input))
            else:
                ticket_dict[department_value] = [(ticket_time, user_input)]

            # Update the session state
            st.session_state['tickets'] = ticket_dict


if __name__ == '__main__':
    main()