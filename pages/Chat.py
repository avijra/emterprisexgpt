import sys
# Insert functions path into working dir if they are not in the same working dir
sys.path.insert(1, "/root/hana_once_again/enterprise/") # Edit <p2_functions folder path> and put path
from pathlib import Path
import torch
import subprocess
import streamlit as st
from run_chat import load_model
from ingest import load_ingest,del_collections
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME, SOURCE_DIRECTORY
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
#from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import emoji
import os
import shutil
from streamlit_extras.stateful_chat import chat, add_message
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.app_logo import add_logo
from prompt_template_utils import get_prompt_template
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from home import authenticator,authentication_status
#st.set_page_config(layout="wide")
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])




hashed_passwords = stauth.Hasher(['redhat','openshift']).generate()
print(hashed_passwords)
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
source_dir=f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS--"+st.session_state["username"]
persist_dir = f"{ROOT_DIRECTORY}/DB--"+st.session_state["username"]
src_file_path=f"{ROOT_DIRECTORY}/init/dummy.txt"

if st.session_state["username"]:
   if os.path.exists(source_dir):
      print("user folder exists")
      
   else:  
       os.makedirs(source_dir)
       try:
        shutil.copy2(src_file_path, source_dir)
        print(f"File copied successfully to {source_dir}")
       except IOError as e:
        print(f"Unable to copy file. {e}")
       except:
        print(f"Unexpected error: {sys.exc_info()}")

#st.set_page_config(layout="wide")

print(MODEL_BASENAME,MODEL_ID)
    


DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

def load_results():
    result=load_ingest(device_type=DEVICE_TYPE)
        #print("this result  "+result)
    st.session_state.result = result


def initialize_session_result_state():
    if "result" not in st.session_state:
     load_results()
    # Run the document ingestion process. 
     if st.session_state["username"]:
        if os.path.exists(persist_dir):
            try:
                print("The directory  exist , making a fresh collections")
                del_collections(st.session_state.result)
                load_results()
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
        else:
            print("The directory does not exist")
            load_results()
            
        
        

# Define the retreiver
# load the vectorstore

        if "EMBEDDINGS" not in st.session_state:
           print("building embeddings")
        EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
        st.session_state.EMBEDDINGS = EMBEDDINGS

        if "DB" not in st.session_state:
            print("building DB")
        DB = Chroma(persist_directory=persist_dir,embedding_function=st.session_state.EMBEDDINGS,client_settings=CHROMA_SETTINGS)
        if DB is None:
            print("Failed to initialize DB in initialize_session_db_state")
        st.session_state.DB = DB
        

   
 
def initialize_session_qa_state():
    if "RETRIEVER" not in st.session_state:
        db = st.session_state.get('DB')
        RETRIEVER = db.as_retriever()
        st.session_state.RETRIEVER = RETRIEVER

    if "LLM" not in st.session_state:
        LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
        st.session_state["LLM"] = LLM




    if "QA" not in st.session_state:

        prompt, memory = get_prompt_template(promptTemplate_type="mistral", history=True)
       # print("this prompt "+prompt +" This is memory" + memory)

        QA = RetrievalQA.from_chain_type(llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=True, callbacks=callback_manager,chain_type_kwargs={"prompt": prompt, "memory": memory},)
        st.session_state["QA"] = QA

def delete_source_route():
    folder_name = source_dir
    
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
      

    os.makedirs(folder_name)
    


def ingestdocc():
   
        print("delete all keys  ")
        for key in list(st.session_state.keys()):
                  if key not in ("name", "username"):
                      del st.session_state[key]

        print(st.session_state.keys()) 
        #st.session_state.username=username   
        initialize_session_result_state()
        initialize_session_qa_state()
        print("RAN")
       
# Sidebar contents

   
with st.sidebar:
         #initialize_session_result_state()
         #initialize_session_qa_state()
         if "result" not in st.session_state: 
           ingestdocc()

         st.write(f'Welcome *{st.session_state["name"]}*')
        # authenticator.logout("Logout", "sidebar")
         if authentication_status == None:
             for key in list(st.session_state.keys()):
                  del st.session_state[key]
         print("delete all keys after logout ")
         
         
         st.title(':red[_EnterPrise X GPT_]')
        

    #uploaded_files = st.file_uploader("Upload your Document", accept_multiple_files=True,on_change= delete_source_route() )
         uploaded_files = st.file_uploader("Upload your Document", accept_multiple_files=True ) 
         if st.button('Delete Documents', help="click me to delete the documents you uploaded "):
      
          delete_source_route() 
          st.toast("Documents sucessfully Deleted.")
   
         if st.button('Create Brain ⚔️', help="click me to create a context for AI with documents you uploaded ",):
          with st.status("creating brain. please wait"):
             ingestdocc() 
             st.toast("New Brain Created. Now please start the Conversation with your Documents")

        # if st.button(" check user session after inject "):
         #   st.write(f'Welcome *{st.session_state["name"]}*')
    
    
for uploaded_file in uploaded_files:
           string = uploaded_file.read()
           with open(os.path.join(source_dir,uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
    
   
            st.write(":red[_File name:_]", uploaded_file.name)
   # st.toast("Please click on Create Brain to create a AI context",icon="⚔️")
    

with chat(key="my_chat"):
   
#prompt = st.text_input('Input your prompt here')
# while True:
      
        if prompt:= st.chat_input():
           add_message("user", prompt, avatar="🧑‍💻")
    # Then pass the prompt to the LLM
    
           response = st.session_state["QA"](prompt)
           answer, docs = response["result"], response["source_documents"]
           add_message("assistant", " AI: ", answer, avatar="⚔️")
           
        
    # ...and write it out to the screen
    #st.write(answer)
    

    # With a streamlit expander  

