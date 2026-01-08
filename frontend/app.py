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
from langchain_core.messages import HumanMessage, AIMessage
from backend.agents import summary_agent
from backend.agents import AgentState
from backend.agents import build_graph



# -----------------------------------------------------------------
# 🧩 PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("🤖 AI Data Analysis Agent")

# Initialize session state


# -----------------------------------------------------------------
# 🧩 Files upload Box
# -----------------------------------------------------------------

# -------------------------
# Tabs layout
# -------------------------
tab1, tab2, tab3 = st.tabs(["📂 Data Management", "💬 Chat Interface", "🧰 Debug Information"])

output_json = {}

with tab1:
    uploaded_file = st.file_uploader("Upload your CSV/Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
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
                st.session_state["schema_summary"] = output_json
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
                        # store schema summary in session state for reuse
                        st.session_state["schema_summary"] = output_json
                        # display it
                        st.write(output_json)
    
                    except Exception as e:
                        st.error(f"Error during summarization: {e}")
           
    # Initialize AgentState        
    def initial_state(schema: dict):
        return {
            "messages": [],
            "schema": schema,
            "needs_code": None,
            "plan": None,
            "code": None,
            "execution_result": None,
            "final_answer": None,
            "trace": []
        }   
with tab2:
    st.subheader("Chat with your Data")
    if "schema_summary" in st.session_state:
        output_json = st.session_state["schema_summary"]
    else:
        st.warning("Please upload a CSV in Tab 1 first.")
        
    if "agent_state" not in st.session_state:
        st.session_state["agent_state"] = initial_state(output_json)
    
    if "agent_app" not in st.session_state:
        st.session_state["agent_app"] = build_graph()
    
        # Render chat history FIRST
    for msg in st.session_state.agent_state["messages"]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
    
    user_input = st.text_input("Enter your question about the dataset:")
    
    if user_input:
       # Append to Agent memory
        st.session_state.agent_state["messages"].append(
            HumanMessage(content=user_input)
            )
        # Invoke the agent
        with st.spinner("Thinking..."):
            st.session_state.agent_state = st.session_state.agent_app.invoke(st.session_state.agent_state)
        
        st.rerun() 
        
    # --- Render the trace panel ---
    with st.expander("🧩 Agent Trace"):
        for event in st.session_state.agent_state.get("trace", []):
            node = event["node"]
            st.markdown(f"**Node:** {node}")
            for k, v in event.items():
                if k != "node":
                    st.markdown(f"- **{k}:** `{v}`")
            st.markdown("---")