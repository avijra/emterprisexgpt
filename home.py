import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import sys
# Insert functions path into working dir if they are not in the same working dir


st.set_page_config(page_title="Beatrix Kiddo", page_icon=":cross_swords:", layout="wide")
hide_bar= """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        visibility:hidden;
        width: 0px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        visibility:hidden;
    }
    </style>
"""

with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
name, authentication_status, username= authenticator.login(':red[_Login to EnterPrise X GPT_]', 'main')

if authentication_status == False:
    st.error("Username/password is incorrect")
    st.markdown(hide_bar, unsafe_allow_html=True)

if authentication_status == None:
    st.warning("Please enter your username and password")
    st.markdown(hide_bar, unsafe_allow_html=True)

if authentication_status:
    
    # # ---- SIDEBAR ----
    st.sidebar.title(f"Welcome {name}")
    # st.sidebar.header("select page here :")
    st.write("# Welcome to EnterPrise X GPT  ")
    
    
    
    
    st.markdown(
        """
    Welcome to EnterPrise X GPT, your solution for cutting-edge, AI-powered document interaction within a secure, multi-tenant environment. Designed specifically for enterprises that require robust privacy and security, Enterprise GPT offers a revolutionary way to interact with and manage your documents. Each user in your organization logs in to a personalized space where they can engage with their data and documents seamlessly, powered by the advanced capabilities of Opensource LLMs like Mistral and LLAMA2 .

    With EnterPrise X GPT, we ensure that every interaction is insulated within the user's domain, making certain that the data privacy is upheld and each user's conversations remain exclusive to their data. Whether it's generating reports, automating functions, or querying datasets, our users can do so with the confidence that they're operating within a secure and segregated workspace.

   
    """
    )

    st.markdown(
    
        """
    Built Using    
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Openshift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)
    - [Mistral AI](https://mistral.ai/)
   
    """
    )

    st.write("Developed by [Abhishek Vijra](https://www.linkedin.com/in/avijra/)")

    

    st.sidebar.success("Select a page above.")
    st.sidebar.markdown(
        """
    - To chat with your Documents click on Chat
    - To see your Uploaded  Documents click on My Documents 
   
   
    """
    )

    ###---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)


    authenticator.logout("Logout", "sidebar")
if authentication_status == None:
    for key in list(st.session_state.keys()):
                  del st.session_state[key]
    print("delete all keys after logout ")              