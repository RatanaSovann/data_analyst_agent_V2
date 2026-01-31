import sys, os, io, re, json, operator
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Annotated

import pandas as pd
import matplotlib as mpl
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from backend.tools import data_view
from backend.detectors.structure_detector import SmartStructureDetector
from backend.utils.fontbank import configure_fontbank
from backend.utils.helper import _safe_json_loads, load_llm_json
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
    file_path: Optional[str] = None
    schema: dict = Field(default_factory=dict) # from summary agent contains file_path
    table_spec: Dict[str, Any] = Field(default_factory=dict)

    # planner output
    next: Optional[str] = None   # "codegen" | "plot_codegen" | "schema_answer"
    plan: Optional[str] = None
    table_handling: Dict[str, Any] = Field(default_factory=dict)

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



# ============================================================================
# 
#   Block 1: Generate table_spec and schema from raw file
# 
#   UNIVERSAL DATA SCHEMA ANALYZER
#
# ============================================================================

class UniversalDataAnalyzer:
    """
    Analyzes a dataset by reading directly from file_path.
    Uses table_spec (from StructureDetector) to read the table correctly and generate schema.
    """

    def __init__(
        self,
        file_path: str,
        table_spec: Dict[str, Any],
        llm: Optional[ChatOpenAI] = None,
    ):
        self.file_path = file_path
        self.table_spec = table_spec or {}
        self.llm = llm or ChatOpenAI(model="gpt-5-nano", temperature=0)

        self.df: Optional[pd.DataFrame] = None
        self.columns: List[str] = []
        self.column_metadata: Dict[str, Any] = {}
        self.data_type: Optional[str] = None

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------
    def analyze(self) -> Dict[str, Any]:
        # 1) Read data correctly using table_spec
        self.df = self._read_dataframe()
        self.columns = [str(c).strip() for c in self.df.columns]

        # 2) Analyze columns
        self.column_metadata = self._analyze_columns()

        # 3) Dataset purpose
        dataset_info = self._understand_dataset()

        # 4) Field semantics
        field_semantics = self._identify_semantics()

        return {
            "data_type": self.data_type,
            "domain": dataset_info.get("domain"),
            "description": dataset_info.get("description"),
            "column_metadata": self.column_metadata,
            "field_semantics": field_semantics
        }

    # ------------------------------------------------------------------
    # Reading logic (table_spec is law)
    # ------------------------------------------------------------------
    def _read_dataframe(self) -> pd.DataFrame:
        skip_rows_top = self.table_spec.get("table_region", {}).get("data_start_row_idx", [])

        df = pd.read_csv(self.file_path, skiprows=skip_rows_top)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Column analysis
    # ------------------------------------------------------------------
    def _analyze_columns(self) -> Dict[str, Dict]:
        metadata: Dict[str, Dict] = {}

        for col in self.columns:
            col_data = self.df[col]

            metadata[col] = {
                "dtype": str(col_data.dtype),
                "content_type": self._infer_content_type(col_data),
                "null_percentage": float(col_data.isna().sum() / len(col_data) * 100),
                "unique_count": int(col_data.nunique(dropna=True)),
                "sample_values": col_data.dropna().head(3).tolist()
                if not col_data.dropna().empty else [],
            }

            if metadata[col]["content_type"] in ["numeric", "currency"]:
                numeric_data = pd.to_numeric(col_data, errors="coerce")
                metadata[col]["statistics"] = {
                    "min": float(numeric_data.min()) if pd.notna(numeric_data.min()) else None,
                    "max": float(numeric_data.max()) if pd.notna(numeric_data.max()) else None,
                    "mean": float(numeric_data.mean()) if pd.notna(numeric_data.mean()) else None,
                }

        return metadata

    def _infer_content_type(self, series: pd.Series) -> str:
        non_null = series.dropna()
        if non_null.empty:
            return "empty"

        try:
            numeric = pd.to_numeric(non_null, errors="coerce")
            if numeric.notna().sum() / len(non_null) > 0.8:
                if any(
                    isinstance(v, str) and any(c in str(v) for c in ["$", "€", "¥", "£"])
                    for v in non_null.head(10)
                ):
                    return "currency"
                return "numeric"
        except Exception:
            pass

        sample = non_null.head(20).astype(str)
        if any(re.search(r"\d{4}-\d{2}-\d{2}", v) or
               re.search(r"\d{2}/\d{2}/\d{4}", v)
               for v in sample):
            return "date"

        uniq_ratio = non_null.nunique() / len(non_null)
        if uniq_ratio > 0.95:
            return "identifier"
        if non_null.nunique() < 50:
            return "categorical"

        if non_null.astype(str).str.len().mean() > 50:
            return "text"

        return "string"

    # ------------------------------------------------------------------
    # LLM: dataset understanding
    # ------------------------------------------------------------------
    def _understand_dataset(self) -> Dict[str, Any]:
        summary = {
            "columns": self.columns,
            "row_count": len(self.df),
            "column_types": {c: m["content_type"] for c, m in self.column_metadata.items()}
        }

        system_prompt = SystemMessage(content="""
Analyze this dataset and determine:
1. What type of data is this? (financial, customer, sensor, inventory, HR, etc.)
2. What domain? (finance, retail, healthcare, IoT, etc.)
3. Brief description of the dataset's purpose

Return JSON:
{
  "data_type": "specific type",
  "domain": "industry domain",
  "description": "1-2 sentence description"
}
""".strip())

        user_prompt = HumanMessage(
            content="Dataset info:\n" + json.dumps(summary, indent=2, ensure_ascii=False)
        )

        response = self.llm.invoke([system_prompt, user_prompt])
        result = _safe_json_loads(response.content)

        self.data_type = result.get("data_type", "unknown")
        return result or {
            "data_type": "unknown",
            "domain": "unknown",
            "description": "Unknown dataset",
        }

    # ------------------------------------------------------------------
    # LLM: field semantics
    # ------------------------------------------------------------------
    def _identify_semantics(self) -> Dict[str, Any]:
        system_prompt = SystemMessage(content="""
You're analyzing a dataset. Add language in the note section
REQUIREMENTS:
- Output MUST be valid JSON only.
- Output MUST contain a single top-level key: "field_semantics".
- Each column name MUST appear exactly once inside "field_semantics".
- Do NOT invent or omit columns.

FIELD SCHEMA:
{
  "semantic_role": [string, ...],
  "business_meaning": string,
  "recommended_data_type": one of ["integer","float","string","date","datetime","boolean"],
  "notes": string
}

Return JSON only in this exact structure:
{
  "field_semantics": {
    "<column_name>": {
      "semantic_role": [],
      "business_meaning": "",
      "recommended_data_type": "string",
      "notes": ""
    }
  }
}
""".strip())

        column_info = [
            {"name": col, "type": meta["content_type"], "samples": meta["sample_values"]}
            for col, meta in self.column_metadata.items()
        ]

        user_payload = {
            "data_type_hint": self.data_type,
            "columns": column_info
        }

        user_prompt = HumanMessage(
            content=json.dumps(user_payload, indent=2, ensure_ascii=False)
        )

        response = self.llm.invoke([system_prompt, user_prompt])
        parsed = _safe_json_loads(response.content)

        if "field_semantics" not in parsed:
            parsed = {"field_semantics": parsed} if parsed else {"field_semantics": {}}

        return parsed


# =========================
# Backend extraction logic for frontend app
# =========================
def compute_table_spec_and_schema(file_path: str, llm):
    detector = SmartStructureDetector(file_path=file_path, llm=llm)
    table_spec = detector.detect_complete_structure()

    analyzer = UniversalDataAnalyzer(
        file_path=file_path,
        table_spec=table_spec,
        llm=llm,
    )
    schema = analyzer.analyze()

    return table_spec, schema

#-------------------------------

# NEW Agent: Summarize dataset structure and content understand headers and merged cell, generate table_spec and schema

# -------------------------------

def structure_and_analysis_node(state: AgentState):
    table_spec, schema = compute_table_spec_and_schema(
        file_path=state.file_path,
        llm=llm
    )

    state.table_spec = table_spec
    state.schema = schema

    state.trace.append({
        "node": "structure+analysis",
        "table_region": table_spec.get("table_region"),
        "data_type": schema.get("data_type"),
        "domain": schema.get("domain"),
    })

    return state


#-------------------------------

# BLOCK 2: FROM the JSON file generated in BLOCK 1, create a flow to analyze the dataset based on user questions

# -------------------------------


def planner_node(state: AgentState):
    schema = state.schema
    table_spec = getattr(state, "table_spec", {}) or {}
    user_msg = state.messages[-1].content


    prompt = f"""
You are a senior data analyst and task router inside an automated data analysis system.

You will be given:
1) Dataset schema: domain, column metadata, and field semantics.
2) Table structure (table_spec): which rows are headers, data, and footers.
3) A user question that may be written in any language.

Your goals:
A) Choose the correct action for the next step.
B) Produce a concrete coding plan that the code generator can follow reliably.
C) Ensure the plan is SAFE for table artifacts: title rows, multi-row headers, and footer totals.
D) Be language-aware: match the user's language in your JSON plan fields, and preserve multilingual dataset content.

=====================
INPUTS
=====================

Dataset schema (columns/types/summary/semantics):
{json.dumps(schema, indent=2, ensure_ascii=False)}

Table structure (table_spec):
{json.dumps(table_spec, indent=2, ensure_ascii=False)}

User question:
"{user_msg}"

=====================
ACTIONS (choose exactly one)
=====================
- "schema_answer": user asks about dataset structure/columns/meaning/header/footer/table layout.
- "plot_codegen": user asks to visualize/plot/chart/graph.
- "codegen": user asks to compute/analyze/filter/aggregate/transform/derive results.

Routing rules:
- If user asks for BOTH analysis and plotting -> choose "plot_codegen".
- If user asks to visualize -> "plot_codegen".
- If user asks about columns/schema/header/footer -> "schema_answer".
- If user asks for a numeric result (totals, net income value, sum, average) -> choose "codegen".
- Otherwise -> "codegen".

=====================
CRITICAL TABLE RULES (MUST FOLLOW)
=====================
1) Always respect table_spec.table_region when planning any computation.
   - header_rows_idx are NOT data and must be excluded from calculations.
   - footer_rows_idx are NOT data and must be excluded from calculations.
   - Use data_start_row_idx and data_end_row_idx to slice the true data region.
   - If footer rows overlap with data_end_row_idx, prioritize footer exclusion.

2) Your plan MUST explicitly describe the row filtering strategy in code terms.
   Preferred approach:
   - df_data = df.iloc[data_start_row_idx : data_end_row_idx]
   - df_data = df_data.drop(index=footer_rows_idx, errors="ignore")
   State that this must happen BEFORE any aggregation.

3) If footer_hints suggests totals or net income (e.g., "Total", "Net Income"):
   - Your plan SHOULD include an optional validation step:
     compute totals from df_data and compare against footer values
     IF footer numeric values are present / parseable.

=====================
DATA TYPE / CLEANING RULES (MUST INCLUDE WHEN NEEDED)
=====================
When the question involves money, totals, net income, aggregations, or comparisons:
- Convert currency strings to numeric:
  - remove currency symbols, commas, whitespace
  - handle parentheses as negatives if present (e.g., "(1,234.00)" -> -1234.0)
- Treat Income and Expense as numeric floats (missing values -> 0).
- Parse DATE into datetime only if time grouping/filtering is needed.
- Preserve multilingual text (e.g., Khmer) in Supplier/DESCRIPTION; do not coerce encoding.

=====================
OUTPUT FORMAT (STRICT)
=====================
Respond ONLY in valid JSON with keys:
- "action": one of ["schema_answer", "codegen", "plot_codegen"]
- "language": a short language label inferred from the user question (e.g., "en", "km", "ja")
- "plan": short concrete plan (written in the user's language)
- "table_handling": object describing how to slice/exclude rows using table_spec

Example output:
{{
  "action": "codegen",
  "language": "en",
  "plan": "Slice to data rows using table_spec; drop footer rows; parse Income/Expense to numeric; compute net income (sum income - sum expense) by Dept.; return a sorted table.",
  "table_handling": {{
    "data_slice": "df.iloc[data_start_row_idx:data_end_row_idx]",
    "drop_rows": "footer_rows_idx",
    "notes": "Exclude header_rows_idx and footer_rows_idx from all aggregations."
  }}
}}
""".strip()

    response = llm.invoke([HumanMessage(content=prompt)])

    # robust-ish parse (optional, but recommended)
    decision = load_llm_json(response.content)

    state.next = decision["action"]
    state.plan = decision["plan"]
    state.table_handling = decision.get("table_handling", {})

    state.trace.append({
        "node": "planner",
        "user_msg": user_msg,
        "action": state.next,
        "plan": state.plan,
        "used_table_spec": True
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
    schema = state.schema
    table_spec = getattr(state, "table_spec", {}) or {}
    table_handling = state.table_handling or {}

    # Small, token-safe context window
    prev_user_msgs = [
        m.content for m in state.messages[-5:]
        if getattr(m, "type", "") in ("human", "user") or m.__class__.__name__ == "HumanMessage"
    ]
    prev_context = {
        "recent_user_messages": prev_user_msgs,
        "previous_code": state.code,
        "previous_execution_result": state.execution_result if state.execution_result else None,
        "previous_final_answer": state.final_answer if state.final_answer else None,
    }
    
    prompt = f"""
You are a senior data scientist writing pandas code inside an automated analysis system.

=====================
RECENT CONTEXT
=====================
{json.dumps(prev_context, indent=2, ensure_ascii=False)}

=====================
SCHEMA (SEMANTICS)
=====================
{json.dumps(schema, indent=2, ensure_ascii=False)}

=====================
TABLE STRUCTURE (SOURCE OF TRUTH)
=====================
table_spec:
{json.dumps(table_spec, indent=2, ensure_ascii=False)}

table_handling (planner output; must follow):
{json.dumps(table_handling, indent=2, ensure_ascii=False)}

Analysis plan:
{plan}

=====================
FOLLOW-UP vs NEW TASK
=====================
Decide if this is a FOLLOW-UP or NEW TASK.
- FOLLOW-UP: tweak filters/grouping/metrics based on prior results.
- NEW TASK: unrelated.

If FOLLOW-UP and previous_code exists: minimally modify previous_code.
If NEW TASK: write fresh code.

=====================
HARD CONSTRAINTS (MUST FOLLOW)
=====================
1) file_path variable is available. Load df from file_path.
2) ABSOLUTELY DO NOT auto-detect headers/footers or scan rows to "find header".
   - You MUST use table_spec.table_region (or table_handling) for row slicing.
   - Only if table_spec/table_handling is missing or empty may you fallback to detection.
3) You MUST exclude:
   - header_rows_idx
   - footer_rows_idx
   from all computations.
4) Always create a clean working dataframe df_data that contains ONLY true data rows.
5) Money parsing:
   - Income/Expense may be strings like "$3,306.18"
   - Convert to numeric floats safely.
   - Missing values -> 0 for computations.
6) DATE parsing:
   - Parse DATE to datetime only if needed (filter/group by date).
   - Use robust parsing for formats like "1-Jul-25".
7) Assign the final output to a variable named `result`.
8) At the end: print(result)
9) Return ONLY valid Python code (no markdown fences inside the "code" string).

Respond in JSON format exactly as follows:

{{
  "thinking": "Describe your thinking / approach in a few sentences.",
  "code": "```python\\n# Python code here\\n```"
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        code_response = load_llm_json(response.content)
    except Exception:
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
    sandbox = {
               "file_path": state.file_path,
               "pd": pd, 
               "re": re, 
               "unicodedata": __import__("unicodedata"), 
               "Counter": __import__("collections").Counter}

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
    import os, json
    from langchain_core.messages import HumanMessage

    user_msg = state.messages[-1].content
    plan = state.plan
    schema = getattr(state, "schema", {}) or {}
    table_spec = getattr(state, "table_spec", {}) or {}
    table_handling = getattr(state, "table_handling", {}) or {}

    state.plot_turn += 1
    os.makedirs("plots", exist_ok=True)
    filename = os.path.join("plots", f"plot_{state.plot_turn:02d}.png")

    prompt = f"""
You are a data visualization expert generating matplotlib code inside an automated analysis system.

=====================
USER QUESTION
=====================
{user_msg}

=====================
SCHEMA (SEMANTICS)
=====================
{json.dumps(schema, indent=2, ensure_ascii=False)}

=====================
TABLE STRUCTURE (SOURCE OF TRUTH)
=====================
table_spec:
{json.dumps(table_spec, indent=2, ensure_ascii=False)}

table_handling (planner output; must follow):
{json.dumps(table_handling, indent=2, ensure_ascii=False)}

=====================
PLOTTING PLAN
=====================
{plan}

=====================
HARD CONSTRAINTS (MUST FOLLOW)
=====================
1) file_path variable is available. Load df from file_path.
2) ABSOLUTELY DO NOT auto-detect headers/footers or scan rows to "find header".
   - You MUST use table_spec.table_region (or table_handling) for row slicing.
   - Only if table_spec is missing/empty may you fallback to detection.
3) You MUST exclude header_rows_idx and footer_rows_idx from plotting/aggregation.
4) Create df_data that contains ONLY true data rows before any plot aggregation.
5) Use matplotlib (seaborn optional, but render via matplotlib). DO NOT use plotly.
6) Do NOT print anything. Do NOT call plt.show().
7) Save exactly to: {filename}
   Use:
     plt.tight_layout()
     plt.savefig("{filename}")
     plt.close()

=====================
RECOMMENDED DATA PREP PATTERN (USE THIS)
=====================
A) Read dataframe:
   - choose pd.read_csv or pd.read_excel based on file extension.

B) Apply table slicing:
   region = table_spec.get("table_region", {{}})
   start = region.get("data_start_row_idx")
   end = region.get("data_end_row_idx")
   footer = region.get("footer_rows_idx") or []
   header = region.get("header_rows_idx") or []

   df_data = df
   if start is not None and end is not None:
       df_data = df.iloc[start:end].copy()

   df_data = df_data.drop(index=[i for i in footer if i in df_data.index], errors="ignore")
   df_data = df_data.drop(index=[i for i in header if i in df_data.index], errors="ignore")

C) If plotting Income/Expense amounts:
   - Convert currency strings like "$3,306.18" to floats.
   - Missing values -> 0.

D) Choose an appropriate chart type automatically based on the question:
   - Trend over time -> line plot
   - Compare categories -> bar chart
   - Distribution -> histogram/box
   - Relationship between two numeric vars -> scatter

=====================
OUTPUT JSON (STRICT)
=====================
Return JSON only (no markdown, no code fences):
{{
  "thinking": "...",
  "code": "import ...\\n...\\nplt.savefig(\\"{filename}\\")\\nplt.close()"
}}
""".strip()

    response = llm.invoke([HumanMessage(content=prompt)])

    # Use your robust parser if available (recommended)
    try:
        code_response = load_llm_json(response.content)
    except Exception:
        code_response = json.loads(response.content)

    state.thinking = code_response["thinking"]
    state.plot_code = code_response["code"]  # no code fences expected now

    state.trace.append({
        "node": "plot_codegen",
        "thinking": state.thinking,
        "code": state.plot_code,
        "plot_filename": filename,
        "used_table_spec": True
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
            "file_path": state.file_path,
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


graph = build_graph()
