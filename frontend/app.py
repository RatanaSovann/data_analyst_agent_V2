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
           

with tab2:
    st.subheader("Chat with your Data")

    if "schema_summary" not in st.session_state:
        st.warning("Please upload a CSV in Tab 1 first.")
        st.stop()

    output_json = st.session_state["schema_summary"]

    if "agent_state" not in st.session_state:
        st.session_state["agent_state"] = AgentState(schema=output_json)
    if "agent_app" not in st.session_state:
        st.session_state["agent_app"] = build_graph()

    state: AgentState = st.session_state.agent_state

    chat_container = st.container()

    # Render past messages
    with chat_container:
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)

    user_input = st.chat_input("Enter your question about the dataset:")

    if user_input:
        # Append human message
        new_messages = state.messages + [HumanMessage(content=user_input)]
        state = state.model_copy(update={"messages": new_messages})
        st.session_state.agent_state = state

        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)

        # Invoke agent
        with st.spinner("Thinking..."):
            result = st.session_state.agent_app.invoke(state)
            state = state.model_copy(update=result if isinstance(result, dict) else result)
            st.session_state.agent_state = state

        # Append AIMessage from final_answer
        if state.final_answer:
            ai_message = AIMessage(content=state.final_answer)
            new_messages = state.messages + [ai_message]
            state = state.model_copy(update={"messages": new_messages})
            st.session_state.agent_state = state

            # Display AI response
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(state.final_answer)


with tab3:
    st.subheader("🧰 Debug / Trace Information")

    if "agent_state" not in st.session_state:
        st.info("No agent state yet. Upload data in Tab 1 and chat in Tab 2 first.")
        st.stop()

    state: AgentState = st.session_state.agent_state

    if not state.trace:
        st.info("No trace events yet. Interact with the agent in Tab 2 to see trace here.")
        st.stop()

    for idx, event in enumerate(state.trace, 1):
        node = event.get("node", "Unknown")
        st.markdown(f"### Event {idx}: Node `{node}`")

        for k, v in event.items():
            if k == "node":
                continue

            text = str(v)

            # Use Python code block for multi-line text or code/comments
            if "\n" in text or "#" in text:
                st.markdown(f"```python\n{text}\n```")
            else:
                st.markdown(f"- **{k}:** `{text}`")

        st.markdown("---")