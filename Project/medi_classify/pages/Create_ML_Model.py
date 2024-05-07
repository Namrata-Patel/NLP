import streamlit as st
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib

from utils.utils import *


if 'cleaned_data' not in st.session_state:
    st.session_state['cleaned_data'] =''
if 'sentences_train' not in st.session_state:
    st.session_state['sentences_train'] =''
if 'sentences_test' not in st.session_state:
    st.session_state['sentences_test'] =''
if 'labels_train' not in st.session_state:
    st.session_state['labels_train'] =''
if 'labels_test' not in st.session_state:
    st.session_state['labels_test'] =''
if 'svm_classifier' not in st.session_state:
    st.session_state['svm_classifier'] =''

 
st.title("Let's build our Model...")
 
# Create tabs
tab_titles = ['Data Preprocessing', 'Model Training', 'Model Evaluation',"Save Model"]
tabs = st.tabs(tab_titles)

# Adding content to each tab

#Data Preprocessing TAB...
with tabs[0]:
    st.header('Data Preprocessing')
    st.write('Here we preprocess the data...')

    # Capture the CSV file
    data = st.file_uploader("Upload CSV file",type="csv")

    button = st.button("Load data",key="data")

    if button:
        with st.spinner('Wait for it...'):
            our_data=read_data(data)
            embeddings=get_embeddings()
            st.session_state['cleaned_data'] = create_embeddings_model(our_data,embeddings)
        st.success('Done!')


#Model Training TAB
with tabs[1]:
    st.header('Model Training')
    st.write('Here we train the model...')
    button = st.button("Train model",key="model")
    
    if button:
            with st.spinner('Wait for it...'):
                st.session_state['sentences_train'], st.session_state['sentences_test'], st.session_state['labels_train'], st.session_state['labels_test']=split_train_test__data(st.session_state['cleaned_data'])
                
                # Initialize a support vector machine, with class_weight='balanced' because 
                # our training set has roughly an equal amount of positive and negative 
                # sentiment sentences
                st.session_state['svm_classifier']  = make_pipeline(StandardScaler(), SVC(class_weight='balanced')) 

                # fit the support vector machine
                st.session_state['svm_classifier'].fit(st.session_state['sentences_train'], st.session_state['labels_train'])
            st.success('Done!')

#Model Evaluation TAB
with tabs[2]:
    st.header('Model Evaluation')
    st.write('Here we evaluate the model...')
    button = st.button("Evaluate model",key="Evaluation")

    if button:
        with st.spinner('Wait for it...'):
            y_pred = st.session_state['svm_classifier'].predict(st.session_state['sentences_test'])
            accuracy_score_val = accuracy_score(st.session_state['labels_test'], y_pred)
            precision_val = precision_score(st.session_state['labels_test'], y_pred, average='weighted')
            recall_val = recall_score(st.session_state['labels_test'], y_pred, average='weighted')
            f1_val = f1_score(st.session_state['labels_test'], y_pred, average='weighted')

            st.success(f"Validation accuracy of svm_classifier : {100*accuracy_score_val:.2f}%")
            st.write(f"Precision: {precision_val}")
            st.write(f"Recall: {recall_val}")
            st.write(f"F1 Score: {f1_val}")

            # st.write("Confusion Matrix:")
            # cm = confusion_matrix(st.session_state['labels_test'], y_pred)
            # st.write(cm)


            st.write("A sample run:")


            #text="lack of communication regarding policy updates salary, can we please look into it?"
            text="Rude driver with scary driving"
            st.write("***Our issue*** : "+text)

            #Converting out TEXT to NUMERICAL representaion
            embeddings= get_embeddings()
            query_result = embeddings.embed_query(text)

            #Sample prediction using our trained model
            result= st.session_state['svm_classifier'].predict([query_result])
            st.write("***Department it belongs to*** : "+result[0])
            

        st.success('Done!')

#Save model TAB
with tabs[3]:
    st.header('Save model')
    st.write('Here we save the model...')

    button = st.button("Save model",key="save")
    if button:

        with st.spinner('Wait for it...'):
             joblib.dump(st.session_state['svm_classifier'], 'modelsvm.pk1')
        st.success('Done!')

