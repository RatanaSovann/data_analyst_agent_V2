# 🧠 CSV Insights Agent Agentic Data Analyst (In Progress)

Turn any CSV into insights, charts, and answers — just by asking questions.

This project is an agentic data analysis system that allows users to upload a dataset and interact with it using natural language. It automatically understands the data structure, generates analysis code, executes it, and returns insights with visualizations.

# 🚀 Features

📂 Upload any CSV file

🧾 Automatic schema detection & data profiling

🤖 Multi-step reasoning using an agent workflow

💬 Ask questions in plain English

📊 Generate charts (bar, line, histogram, etc.)

🧠 Maintains context across conversations

🔍 Handles messy, real-world CSV dataset

# 🏗️ Architecture

- The system is built using an agentic workflow powered by LangGraph:

````md
User Query
   ↓
Planner Node → decides what to do
   ↓
Router
   ├── Schema Answer (simple questions)
   └── Code Generation (complex analysis)
            ↓
        Executor (runs Python code)
            ↓
        Interpreter (explains results)
            ↓
        Final Answer + Visualizations
````

# 🛠️ Tech Stack
- Language: Python
- Frameworks: Streamlit, LangGraph, LangChain
- LLM: OpenAI (GPT models) / Ollama (optional local)
- Data: Pandas, NumPy
- Visualization: Matplotlib
- Validation: Pydantic (IN PROGRESS)

# ▶️ Usage
User upload their CSV files 
<img width="852" height="437" alt="1" src="https://github.com/user-attachments/assets/dfa7780e-9aba-4fb9-a591-ae0101e3f2dd" />
<img width="1300" height="700" alt="2" src="https://github.com/user-attachments/assets/78348fad-8809-4df1-8fe3-25b2f1f14442" />

- The CSV structure will automatically get detected and stored in memory for Agent to understand 
<img width="1600" height="1000" alt="3" src="https://github.com/user-attachments/assets/47704424-3d99-43ed-96d3-63b1b4ed70ce" />
<img width="700" height="700" alt="4" src="https://github.com/user-attachments/assets/f162c5d6-defa-46f8-b8bb-dd557eb671f8" />

- Semantics meaning of each field are also generated so that agent can understand what each column means. The idea is Data Analyst, can just infer meaning from just observing what each cell contains. It also helps provide context for the LLM when planning & performing analysis. Language semantics is also included in the prompt to support multi-liguistic analysis.

<img width="1100" height="1000" alt="field semantics" src="https://github.com/user-attachments/assets/532e0bd0-f559-4deb-8a93-005de5536364"/>
<img width="1200" height="1200" alt="language field semantics" src="https://github.com/user-attachments/assets/a293fe72-39c2-4cd0-82ff-0c9ac63648f9" />


In the Chat Interface Tab user can interact with the data conversationally: 

<img width="1000" height="600" alt="chat example" src="https://github.com/user-attachments/assets/31ea59d6-def8-45c3-b623-886520ecb3f7" />
<img width="1000" height="600" alt="chat example answer" src="https://github.com/user-attachments/assets/2c291aa9-7320-4e3f-82de-e415b092a4b1" />

- There is a third tab for debugging to see the behind the scene on how agent reach the answer

 <img width="1400" height="600" alt="debug 1" src="https://github.com/user-attachments/assets/0ecbdf01-3279-48e2-8575-4e30c7553e38" />

The planner agent will analyze the question and generate plans on how to retrieve the answer in clear instructions. It will also route to a code agent that generates python code based on the plan. 

<img width="2000" height="1200" alt="debug 2 code gen" src="https://github.com/user-attachments/assets/7f1c3564-4b0f-4ee3-9a0d-e127028561f7" />
<img width="1500" height="400" alt="dewbug 5 answer" src="https://github.com/user-attachments/assets/8bdebc79-7fa4-4aa2-b8a8-64959d8a7817" />






