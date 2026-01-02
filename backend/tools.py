
import pandas as pd
from typing import Annotated, Tuple, List, Dict
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL
import pickle
import os

@tool("data_view")
def data_view(file_name: str) -> str:
    """
    Tool function to read an Excel file from the uploads folder and return a preview.
    """
    try:
        # Construct full file path
        file_to_use = os.path.join(os.getcwd(), "uploads", file_name)
        
        # Read Excel file
        df = pd.read_excel(file_to_use)
        
        # Return preview 
        preview = df.head()
        return f"✅ Loaded `{file_name}` successfully.\n\n**Preview:**\n{preview}"
    
    except Exception as e:
        return f"❌ Error reading file `{file_name}`: {e}"


