import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from typing import Annotated, TypedDict, Sequence, Dict
from langchain_core.messages import BaseMessage
import operator
from tools import data_view
from langchain_core.messages import HumanMessage


os.getcwd
load_dotenv()


llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# leverage Create React Agent to generate summary as the task is straightforward

# -------------------------------
summary_prompt = """You are a concise dataset summarizer.

You will receive a message containing a file path to a dataset.

Steps:
1. Call the `data_view` tool with the exact file path.
2. From the tool output, extract all column names and their data types.
3. Create a description for each column.
4. Output STRICTLY in JSON format with two keys:
   - 'columns': a list of dictionaries with keys 'Column', 'Type', 'Meaning'
   - 'summary': a short paragraph summarizing the dataset

Example output:
{
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

file_path = r"..\uploads\cafe.xlsx"



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

