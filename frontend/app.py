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
from backend.agents import compute_table_spec_and_schema
from backend.agents import AgentState, llm
from backend.agents import build_graph
from backend.utils.helper import load_json, save_json, sha256_file, safe_read_preview, read_raw_table




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
        # ---- Save file ----
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state["file_path"] = file_path

        # ---- Quick preview (as typical dataframe) ----
        st.subheader("Uploaded Data Preview (standard)")
        try:
            df_preview = safe_read_preview(file_path)
            st.dataframe(df_preview.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            st.stop()

        # ---- Cache by file content hash ----
        cache_dir = "dataset_cache"
        os.makedirs(cache_dir, exist_ok=True)

        file_hash = sha256_file(file_path)
        st.session_state["file_hash"] = file_hash

        table_spec_path = os.path.join(cache_dir, f"{file_hash}_table_spec.json")
        schema_path = os.path.join(cache_dir, f"{file_hash}_schema.json")

        # ---- Load or compute structure + schema ----
        if os.path.exists(table_spec_path) and os.path.exists(schema_path):
            st.success("Cached structure + schema found. Loaded from disk.")
            table_spec = load_json(table_spec_path)
            schema = load_json(schema_path)
        else:
            with st.spinner("Extracting structure + analyzing dataset..."):
                try:
                    table_spec, schema = compute_table_spec_and_schema(file_path, llm)
                    save_json(table_spec_path, table_spec)
                    save_json(schema_path, schema)
                except Exception as e:
                    st.error(f"Extraction/analysis failed: {e}")
                    st.stop()
            st.success("Structure + schema extracted and cached.")

        # ---- Store for agent usage ----
        st.session_state["table_spec"] = table_spec
        st.session_state["schema_summary"] = schema

        # =========================
        # STRUCTURE-AWARE PREVIEW (from heading)
        # =========================
        st.subheader("Table Preview (structure-aware)")

        region = (table_spec or {}).get("table_region", {})
        header_rows = region.get("header_rows_idx", []) or []
        data_start = region.get("data_start_row_idx", None)
        data_end = region.get("data_end_row_idx", None)
        footer_rows = region.get("footer_rows_idx", []) or []

        # Read raw table for custom header logic
        try:
            df_raw = read_raw_table(file_path)
        except Exception as e:
            st.warning(f"Could not read raw table for structure preview: {e}")
            df_raw = None

        if df_raw is not None:
            preview_mode = st.radio(
                "Preview mode",
                ["Raw grid (no header)", "From heading (skip title rows)", "Data-only (exclude header/footer)"],
                horizontal=True
            )

            if preview_mode == "Raw grid (no header)":
                st.dataframe(df_raw.head(25))

            elif preview_mode == "From heading (skip title rows)":
                # Header row index logic:
                # Prefer: header row is immediately after title/header rows.
                # If detector sets data_start_row_idx to that header row, use it.
                # If data_start_row_idx is first data row, fallback to max(header_rows)+1 for column names.
                fallback_header_row_idx = (max(header_rows) + 1) if header_rows else 0
                header_row_idx = data_start if data_start is not None else fallback_header_row_idx

                if header_row_idx >= len(df_raw):
                    st.error("Detected header row is outside the table range.")
                else:
                    new_cols = df_raw.iloc[header_row_idx].astype(str).tolist()
                    df_from_heading = df_raw.iloc[header_row_idx + 1 :].copy()
                    df_from_heading.columns = new_cols
                    df_from_heading = df_from_heading.dropna(how="all")

                    st.caption(f"Using row index {header_row_idx} as the heading (column names).")
                    st.dataframe(df_from_heading.head(30))

            else:  # Data-only
                df_data = df_raw.copy()

                if data_start is not None and data_end is not None:
                    df_data = df_raw.iloc[data_start:data_end].copy()

                df_data = df_data.drop(index=[i for i in header_rows if i in df_data.index], errors="ignore")
                df_data = df_data.drop(index=[i for i in footer_rows if i in df_data.index], errors="ignore")
                df_data = df_data.dropna(how="all")

                st.caption(f"Data slice: rows {data_start} to {data_end}. Footer dropped: {footer_rows}")
                st.dataframe(df_data.head(30))

        # =========================
        # DISPLAY ANALYSIS (schema/profile)
        # =========================
        st.subheader("Dataset Analysis")
        st.json(schema)
           

with tab2:
    st.subheader("Chat with your Data")

    # Require upload + extracted artifacts
    if "file_path" not in st.session_state or "schema_summary" not in st.session_state or "table_spec" not in st.session_state:
        st.warning("Please upload a CSV/Excel file in Tab 1 first.")
        st.stop()

    file_path = st.session_state["file_path"]
    schema = st.session_state["schema_summary"]
    table_spec = st.session_state["table_spec"]

    # Init agent + graph once
    if "agent_state" not in st.session_state:
        st.session_state["agent_state"] = AgentState(
            file_path=file_path,
            schema=schema,
            table_spec=table_spec,
            messages=[],
        )
    else:
        # If user uploads a new file, refresh state core fields but keep chat history if you want
        state: AgentState = st.session_state["agent_state"]
        updates = {}
        if getattr(state, "file_path", None) != file_path:
            updates["file_path"] = file_path
            # Optional: clear messages when file changes
            updates["messages"] = []
            # Reset plot bookkeeping
            updates["plot_paths"] = []
            updates["plot_turn"] = 0
        updates["schema"] = schema
        updates["table_spec"] = table_spec
        st.session_state["agent_state"] = state.model_copy(update=updates)

    if "agent_app" not in st.session_state:
        st.session_state["agent_app"] = build_graph()

    state: AgentState = st.session_state["agent_state"]

    # One container for the whole transcript
    chat_container = st.container()

    def render_message(msg):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
                plot_paths = (msg.additional_kwargs or {}).get("plot_paths", [])
                for p in plot_paths:
                    # Streamlit likes use_container_width=True
                    st.image(p, use_container_width=True)

    # Render past messages
    with chat_container:
        for msg in state.messages:
            render_message(msg)

    user_input = st.chat_input("Enter your question about the dataset:")

    if user_input:
        # Append the user message to state.messages
        state = state.model_copy(update={"messages": state.messages + [HumanMessage(content=user_input)]})
        st.session_state["agent_state"] = state

        # Render user message immediately
        with chat_container:
            render_message(state.messages[-1])

        # Snapshot plot count BEFORE invoke (so we can detect new plots)
        prev_n = len(getattr(state, "plot_paths", []) or [])

        # Invoke agent
        with st.spinner("Thinking..."):
            result = st.session_state["agent_app"].invoke(state)

        # Normalize returned state
        if isinstance(result, AgentState):
            state = result
        elif isinstance(result, dict):
            state = state.model_copy(update=result)
        else:
            # fallback: keep state unchanged but show error
            st.error("Agent returned an unexpected result type.")
            st.session_state["agent_state"] = state
            st.stop()

        st.session_state["agent_state"] = state

        # Detect any new plots generated in this turn
        plot_paths_all = getattr(state, "plot_paths", []) or []
        new_plots = plot_paths_all[prev_n:]

        # If graph produced final_answer, append as an AI message to messages
        if getattr(state, "final_answer", None):
            ai_kwargs = {"plot_paths": new_plots} if new_plots else {}
            ai_message = AIMessage(content=state.final_answer, additional_kwargs=ai_kwargs)

            state = state.model_copy(update={"messages": state.messages + [ai_message]})
            st.session_state["agent_state"] = state

            # Render assistant message immediately
            with chat_container:
                render_message(ai_message)
        else:
            # If no final_answer, still show something to the user
            with chat_container:
                with st.chat_message("assistant"):
                    st.write("No final answer was produced for this turn.")
                    for p in new_plots:
                        st.image(p, use_container_width=True)


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