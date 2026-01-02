import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from typing import Annotated, TypedDict, Sequence, Dict
from langchain_core.messages import BaseMessage
import operator
from tools import data_view
from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Optional
import json
import io
import sys
import re
import pandas as pd
from langgraph.graph import StateGraph, END



os.getcwd
load_dotenv()


llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

#-------------------------------

# BLOCK 1: Automatically summarize dataset structure and content when user uploads a file

# -------------------------------
summary_prompt = """You are a concise dataset summarizer. Use only pandas to read the file.

You will receive a message containing a file path to a dataset. 

Steps:
1. Call the `data_view` tool with the exact file path.
2. From the tool output, extract all column names and their data types.
3. Create a description for each column.
4. Output STRICTLY in JSON format with two keys:
   - 'file_path': the original file path
   - 'columns': a list of dictionaries with keys 'Column', 'Type', 'Meaning'
   - 'summary': a short paragraph summarizing the dataset

Example output:

{
  "file_path": "uploads/cafe.xlsx",
  "columns": [
    {"Column": "Date", "Type": "datetime", "Meaning": "Timestamp of the transaction"},
    {"Column": "Receipt number", "Type": "string", "Meaning": "Unique identifier for the transaction"}
  ],
  "summary": "The dataset represents café sales transactions, including timestamps, receipt numbers, sales, discounts, and items purchased."
}
"""

summary_agent = create_agent(
    model=llm,
    tools=[data_view],
    name="SummaryAgent",
    system_prompt=summary_prompt
)

# Test Agent Output

file_path = r"C:\\Users\\sovan\\Desktop\\CSV_analyst_agent\\uploads\\cafe.xlsx"


result = summary_agent.invoke({
    "file_path": file_path,
    "messages": [
        HumanMessage(
            content=f"Summarize this dataset in Streamlit JSON format: {file_path}"
            )
        ],
    })
output_text = result["messages"][-1].content
print(output_text)

# Create a new folder to save agent outputs schema if not exists
output_folder = os.path.join(os.getcwd(), "dataset_summaries")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Extract file name without extension
file_name = os.path.splitext(os.path.basename(file_path))[0]
# Save output to a JSON file
output_file_path = os.path.join(output_folder, f"{file_name}_summary.json")
with open(output_file_path, "w") as f:
    f.write(output_text)


#-------------------------------

# BLOCK 2: FROM the JSON file generated in BLOCK 1, create a flow to analyze the dataset based on user questions

# -------------------------------

# Define Agent State

class AgentState(TypedDict):
    messages: List[BaseMessage]
    schema: dict
    needs_code: Optional[bool]
    plan: Optional[str]
    code: Optional[str]
    execution_result: Optional[str]
    final_answer: Optional[str]
    
    

with open(output_file_path, "r", encoding="utf-8") as f:
    schema = json.load(f)
    
state = {
    "messages": [HumanMessage(content="What was the most popular item sold in July 2023?")],
    "schema": schema
}


# Test planner node

def planner_node(state: AgentState):
    schema = state["schema"]
    user_msg = state["messages"][-1].content

    prompt = f"""
You are a senior data analyst.

Dataset schema:
{json.dumps(schema, indent=2)}

User question:
"{user_msg}"

Decide if code is required.

Respond ONLY in JSON:
{{
  "needs_code": true | false,
  "plan": "short reasoning or analysis plan"
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = json.loads(response.content)

    # Update state in-place
    state["needs_code"] = decision["needs_code"]
    state["plan"] = decision["plan"]

    return state
    

# Test Schema answer node

def schema_answer_node(state: AgentState):
    schema = state["schema"]
    user_msg = state["messages"][-1].content

    prompt = f"""
You are a data analyst. 

- DO not mention file paths or columns details unless specifically asked.
- Answer in one short paragraph.
{json.dumps(schema, indent=2)}

Question:
{user_msg}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Update state
    state["final_answer"] = response.content
    return state

# Code Gen node

# Remove ```python ``` markers
def extract_python(code_str):
    match = re.search(r"```python(.*?)```", code_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code_str.strip()

def codegen_node(state: AgentState):
    user_msg = state["messages"][-1].content
    plan = state["plan"]

    prompt = f"""
You are a data scientist expert.

User question:
{user_msg}

Schema:
{json.dumps(state["schema"], indent=2)}

Analysis plan:
{plan}


Your task:
1. Write a short exploratory analysis reasoning ("thinking") before coding.
2. Then write Python (pandas) code to carry it out.

Constraints:
- Assume dataframe is already loaded as `df` if needed, or you can load from file_path.
- Use clean, readable code.
- Assign final output to a variable `result`.
- At the end of the code, always print result to view it.
- Return **only valid Python code** in the code section.

Respond in JSON format exactly as follows:

{{
  "thinking": "Describe your thinking / approach in a few sentences.",
  "code": "```python\\n# Python code here\\n```"
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    code_response = json.loads(response.content)
    
    # Update state
    state["thinking"] = code_response["thinking"]
    # Remove ```python ``` markers before storing
    state["code"] = extract_python(code_response["code"])

    return state

# Test the executor and its output

def executor_node(state: dict):
    code = state["code"]

    # Sandbox with pre-imported modules
    local_vars = {}
    global_vars = {"pd": pd, "re": re}

    # Capture print statements
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()

    try:
        exec(code, global_vars, local_vars)
        output = mystdout.getvalue()
        result = local_vars.get("result", None)
    except Exception as e:
        output = str(e)
        result = None
    finally:
        sys.stdout = old_stdout

    # Update state
    state["execution_result"] = output.strip()
    state["result"] = result
    return state


# Test final answer node
def interpreter_node(state: AgentState):
    user_msg = state["messages"][-1].content
    output = state["execution_result"]

    prompt = f"""
You are a data analyst explaining results to a stakeholder.

Question:
{user_msg}

Output:
{output}

Explain concisely and clearly the result above in context of the question using simple words. Don not add unecessary details unless asked.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Update state
    state["final_answer"] = response.content
    return state
    

# Routing Logic

def route_after_planner(state: AgentState):
    return "codegen" if state["needs_code"] else "schema_answer"


# --------------------------------

# Define the Workflow Graph

# --------------------------------

graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("schema_answer", schema_answer_node)
graph.add_node("codegen", codegen_node)
graph.add_node("executor", executor_node)
graph.add_node("interpreter", interpreter_node)

graph.set_entry_point("planner")

graph.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "schema_answer": "schema_answer",
        "codegen": "codegen"
    }
)

graph.add_edge("schema_answer", END)
graph.add_edge("codegen", "executor")
graph.add_edge("executor", "interpreter")
graph.add_edge("interpreter", END)

app = graph.compile()

# Execute the Graph



with open(output_file_path, "r", encoding="utf-8") as f:
    schema = json.load(f)


state = {
    "messages": [HumanMessage(content="What is the top 5 most popular items sold in July 2023?")],
    "schema": schema  # your pre-loaded schema JSON
}


result = app.invoke(state)


print(result.get("needs_code"))
print(result.get("plan"))
print(result.get("code"))
print(result.get("execution_result"))
print(result.get("final_answer"))
