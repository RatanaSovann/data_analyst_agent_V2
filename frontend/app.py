import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Set ROOT to the project root directory
sys.path.append(str(ROOT))

import pickle
import streamlit as st
import os
import pandas as pd
import json
from backend.tools import data_view  
from langchain_core.messages import HumanMessage
from backend.agents import summary_agent


# -----------------------------------------------------------------
# 🧩 PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("🤖 AI Data Analysis Agent")

# -----------------------------------------------------------------
# 🧩 Files upload Box
# -----------------------------------------------------------------

os.makedirs("uploads", exist_ok=True)

uploaded_file = st.file_uploader("Upload your CSV/Excel file", type=["csv", "xlsx"])

if uploaded_file:
    file_path = os.path.join(os.getcwd(),"uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.subheader("Uploaded Data Preview")
   
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            st.dataframe(df.head())
            
        else:  
            df = pd.read_csv(file_path)
            st.dataframe(df.head())
    
    except Exception as e:
        st.error(f"Error reading file: {e}")
        
    # Confirm file exists before summarization
    if not os.path.exists(file_path):
        st.error(f"file not found: {file_path}")
        
    else:
        # Check to see if summary already exists, if exists, skip summarization and load JSON file
        output_folder = os.path.join(os.getcwd(), "dataset_summaries")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_path = os.path.join(output_folder, f"{file_name}_summary.json")
        
        if os.path.exists(output_file_path):
            st.success("Summary already exists. Loaded existing summary.")
            output_json = json.load(open(output_file_path, "r"))
            st.write(output_json)
    
        else:
            with st.spinner("Summarizing dataset..."):
                try:
                    result = summary_agent.invoke({
                        "file_path": file_path,
                        "messages": [
                            HumanMessage(
                                content=f"Summarize this dataset in Streamlit JSON format: {file_path}"
                                )
                            ],
                        })
                    
                    output_text = result["messages"][-1].content
                    
                    if os.path.exists(output_folder) is False:
                        os.makedirs(output_folder)
                    
                    with open(output_file_path, "w") as f:
                        f.write(output_text)
                    
                    output_json = json.load(open(output_file_path, "r"))
                    st.write(output_json)

                except Exception as e:
                    st.error(f"Error during summarization: {e}")
                
            
