import streamlit as st
import os
import pandas as pd

def list_files_in_directory(target_directory):
    """ List all files in the given directory """
    files = os.listdir(target_directory)
    return files

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
source_dir=f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS--"+st.session_state["username"]

st.title(':red[_EnterPrise X GPT_]')
st.write('My Documents')

    # Set the target directory to the current directory for the example
    # You can change it to the directory you want to list files from
target_directory = source_dir


        # Get the list of files
files_list = list_files_in_directory(target_directory)

        # Convert the list of files to a Pandas DataFrame
files_df = pd.DataFrame(files_list, columns=['File Name'])

        # Display the DataFrame as a table in Streamlit
st.table(files_df)