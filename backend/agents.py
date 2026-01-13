import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Set ROOT to the project root directory
sys.path.append(str(ROOT))

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from typing import Annotated, Any, TypedDict, Sequence, Dict
from langchain_core.messages import BaseMessage
import operator
from backend.tools import data_view
from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Optional
import json
import io
import re
import pandas as pd
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from backend.utils.fontbank import configure_fontbank
import matplotlib as mpl
load_dotenv()


llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True
)


# Define Agent State

class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    schema: dict = Field(default_factory=dict) # from summary agent contains file_path

    # planner output
    next: Optional[str] = None   # "codegen" | "plot_codegen" | "schema_answer"
    plan: Optional[str] = None

    # reasoning & tracing
    thinking: Optional[str] = None
    trace: List[dict] = Field(default_factory=list)

    # normal code path
    code: Optional[str] = None
    execution_result: Optional[str] = None
    result_var: Optional[Any] = None

    # plot path
    plot_code: Optional[str] = None
    plot_paths: List[str] = Field(default_factory=list)
    plot_turn: int = 0

    # final answer
    final_answer: Optional[str] = None

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


#-------------------------------

# BLOCK 2: FROM the JSON file generated in BLOCK 1, create a flow to analyze the dataset based on user questions

# -------------------------------

    
# Test planner node

def planner_node(state: AgentState):
    schema = state.schema
    user_msg = state.messages[-1].content

    prompt = f"""
You are a senior data analyst and task router inside an automated data analysis system.

Dataset schema:
{json.dumps(schema, indent=2)}

User question:
"{user_msg}"

Your job is to decide what kind of action the system should take.

You must choose EXACTLY ONE of these actions:

- "schema_answer": if the user is asking about the dataset structure, columns, schema, or what fields mean.
- "plot_codegen": if the user is asking to visualize, plot, chart, graph, or draw something.
- "codegen": if the user is asking to compute, analyze, filter, aggregate, transform, or derive results from the data.

Rules:
- If the user asks for BOTH analysis and plotting, choose "plot_codegen".
- If the user asks to visualize, choose "plot_codegen".
- If the user asks about columns or schema, choose "schema_answer".
- Otherwise, choose "codegen".

Respond ONLY in valid JSON with the keys:
- action: one of ["schema_answer", "codegen", "plot_codegen"]
- plan: a short, concrete description of what should be done

Example:
{{
  "action": "plot_codegen",
  "plan": "Plot total sales by month using matplotlib"
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = json.loads(response.content)

    state.next = decision["action"]
    state.plan = decision["plan"]

    state.trace.append({
        "node": "planner",
        "user_msg": user_msg,
        "action": state.next,
        "plan": state.plan
    })

    return state


def schema_answer_node(state: AgentState):
    schema = state.schema
    user_msg = state.messages[-1].content

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
    state.final_answer = response.content
    
    # For tracing/debugging
    state.trace.append({
        "node": "schema_answer",
        "user_msg": state.messages[-1].content,
        "final_answer": state.final_answer
    })
    return state

# Code Gen node

# Remove ```python ``` markers
def extract_python(code_str):
    match = re.search(r"```python(.*?)```", code_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code_str.strip()

def codegen_node(state: AgentState):
    user_msg = state.messages[-1].content
    plan = state.plan

    prompt = f"""
You are a data scientist expert.

User question:
{user_msg}

Schema:
{json.dumps(state.schema, indent=2)}

Analysis plan:
{plan}

Your task:
1. Write a short exploratory analysis reasoning ("thinking") before coding.
2. Then write Python (pandas) code to carry it out.

Constraints:
- Load dataframe from file_path as df.
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
    state.thinking = code_response["thinking"]
    # Remove ```python ``` markers before storing
    state.code = extract_python(code_response["code"])
    
    # For tracing/debugging
    state.trace.append({
        "node": "codegen",
        "thinking": state.thinking,
        "code": state.code
    })

    return state

# Test the executor and its output

def executor_node(state: AgentState):
    code = state.code

    # Use a single dict for both globals and locals
    sandbox = {"pd": pd, "re": re, "unicodedata": __import__("unicodedata"), "Counter": __import__("collections").Counter}

    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()

    try:
        exec(code, sandbox)  # everything now shares same namespace
        result_var = sandbox.get("result", None)
        output = mystdout.getvalue()
    except Exception as e:
        output = str(e)
        result_var = None
    finally:
        sys.stdout = old_stdout

    # Update state
    state.execution_result = output.strip()
    state.result_var = result_var

    state.trace.append({
        "node": "executor",
        "execution_result": state.execution_result,
        "result_var": repr(result_var),
    })

    return state

def plot_codegen_node(state: AgentState):
    user_msg = state.messages[-1].content
    plan = state.plan
    state.plot_turn += 1
    
    filename = os.path.join("plots", f"plot_{state.plot_turn:02d}.png")
    
    prompt = f"""
You are a data visualization expert.

User question:
{user_msg}

Schema:
{json.dumps(state.schema, indent=2)}

Plotting plan:
{plan}

Your task:
1. Write a short explanation ("thinking") of what you will plot and why.
2. Then write Python code matplotlib (or seaborn, but must render via matplotlib)to generate plot. Do not use plotly. 

Constraints:
- load 'df' from file path.
- Do NOT include: if __name__ == "__main__":
- Choose an appropriate chart type automatically (line, bar, scatter, histogram, box, etc).
- Do NOT print anything.
- Do NOT call plt.show().
- Save to filename exactly: {filename}

Saving rules:
- matplotlib/seaborn:
    plt.tight_layout()
    plt.savefig("{filename}")
    plt.close()

- plotly:
    fig.write_image("{filename}")

Return JSON only (no markdown, no code fences):

{{
  "thinking": "Describe what you will plot and why.",
  "code": "```python\\n# Python code here\\n```"
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    code_response = json.loads(response.content)

    # Update state
    state.thinking = code_response["thinking"]
    state.plot_code = extract_python(code_response["code"])

    # For tracing/debugging
    state.trace.append({
        "node": "plot_codegen",
        "thinking": state.thinking,
        "code": state.plot_code
    })

    return state

BANNED = [
    r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:',
    r'\bdef\s+main\s*\(',
    r'\b__name__\b',
]

def plot_executor_node(state: AgentState):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    import re

    # 1) Ensure plots folder exists
    PLOTS_DIR = "plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 2) Use the SAME naming logic as plot_codegen_node
    filename = os.path.join(PLOTS_DIR, f"plot_{state.plot_turn:02d}.png")

    # 3) Safety check (prevents __main / main)
    code = (state.plot_code or "").strip()
    if not code:
        raise ValueError("plot_code is empty")

    for pat in BANNED:
        if re.search(pat, code):
            raise ValueError(f"Rejected plot code (banned pattern): {pat}")
        
    old_rc = mpl.rcParams.copy()
    try:
        configure_fontbank() # Render different language fonts properly
        # Inject plotting libs into exec
        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "px": px,
            "go": go,
            "re": re,
            "filename": filename,      # optional: lets code do plt.savefig(filename)
        }

        # 5) Run the generated plot code
        exec(code, exec_globals, {})
    finally:
        # Restore rcParams to avoid side effects
        mpl.rcParams.update(old_rc)

    # 6) Save path for frontend
    state.plot_paths.append(filename)

    state.final_answer = "Here’s the chart."

    state.trace.append({
        "node": "plot_executor",
        "plot_number": state.plot_turn,
        "plot_path": state.plot_paths,
    })

    return state

# Test final answer node
def interpreter_node(state: AgentState):
    user_msg = state.messages[-1].content
    output = state.execution_result

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
    state.final_answer = response.content
    
    # For tracing/debugging
    state.trace.append({
        "node": "interpreter",
        "final_answer": state.final_answer
    })
    return state
    

# Routing Logic

def route_after_planner(state: AgentState):
    if state.next not in {"codegen", "plot_codegen", "schema_answer"}:
        return "codegen"  # safe default
    return state.next


# --------------------------------

# Define the Workflow Graph

# --------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("schema_answer", schema_answer_node)

    # normal analysis path
    graph.add_node("codegen", codegen_node)
    graph.add_node("executor", executor_node)
    graph.add_node("interpreter", interpreter_node)

    # plot path
    graph.add_node("plot_codegen", plot_codegen_node)
    graph.add_node("plot_executor", plot_executor_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "schema_answer": "schema_answer",
            "codegen": "codegen",
            "plot_codegen": "plot_codegen",
        }
    )

    # schema path
    graph.add_edge("schema_answer", END)

    # normal code path
    graph.add_edge("codegen", "executor")
    graph.add_edge("executor", "interpreter")
    graph.add_edge("interpreter", END)

    # plot path
    graph.add_edge("plot_codegen", "plot_executor")
    graph.add_edge("plot_executor", END)

    return graph.compile()




