import pickle
import streamlit as st
import os
import pandas as pd
import json
from tools import data_view  
from langchain_community.chat_models import ChatOllama
from ollama import Client


# -----------------------------------------------------------------
# 🧩 PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("🤖 AI Data Analysis Agent")

os.makedirs("uploads", exist_ok=True)
os.makedirs("images/plotly_figures/pickle", exist_ok=True)


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])


if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.subheader("Uploaded Data Preview")
   
    try:
        df = pd.read_csv(file_path)
        st.dataframe(df.head())
    
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        
        

file_path = r"C:\Users\sovan\Desktop\CSV_analyst_agent\uploads\cafe.xlsx"
messages = [
  {
    "role": "system",
    "content": 
      """
You are a dataset summarizer.
You have access to the file_path to the dataset

Steps:
1. Call the tool 'data_view' with the file_path to view the dataset.
2. From the tool output, extract all column names and their data types.
3. Create a description for each column.
4. Output STRICTLY in JSON format with four keys:
    - 'tool': "<tool name>"
    - 'arguments': { <tool arguments> or null }
    - 'columns': a list of dictionaries with keys 'Column', 'Type', 'Meaning'
    - 'summary': a short paragraph summarizing the dataset
    
Available tool:
    - data_view(file_name: str) -> str : Load a single Excel file from the given file path and return a preview.
  
"""
  },
  {
    "role": "user",
    "content": "Summarize the dataset located at  " + file_path
  }
]



client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

resp = client.chat(
    model="gpt-oss:120b",
    messages=messages
)

raw_text = resp["message"]["content"]
print(raw_text)


from pydantic import BaseModel
from typing import List, Optional

class ColumnInfo(BaseModel):
    Column: str
    Type: str
    Meaning: str

class DatasetOutput(BaseModel):
    tool: str
    arguments: dict | None = None
    columns: List[ColumnInfo]
    summary: str

TOOLS = {
    "data_view": data_view
}


client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

buffer = ""
for part in client.chat(model="gpt-oss:120b", messages=messages, stream=True):
    
    print(part)
    
    token = part['message']['content']
    print(token, end='', flush=True)   # live streaming
    buffer += token



# Validate JSON output
try:
    agent_output = DatasetOutput.model_validate_json(buffer)
except Exception as e:
    raise RuntimeError("Invalid JSON from LLM") from e

# Execute tool
tool_fn = TOOLS.get(agent_output.tool)
if tool_fn:
    result = tool_fn(**agent_output.arguments)
    print("\n[Tool output]")
    print(result)
