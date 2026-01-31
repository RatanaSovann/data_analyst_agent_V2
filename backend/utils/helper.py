import os
import json
import hashlib
import pandas as pd
import re
# =========================
# Helpers
# =========================
def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_read_preview(file_path: str) -> pd.DataFrame:
    """Quick preview with best-effort encoding for CSV; normal header handling."""
    if file_path.lower().endswith(".xlsx"):
        return pd.read_excel(file_path)
    try:
        return pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="utf-8")

def read_raw_table(file_path: str) -> pd.DataFrame:
    """
    Read as a raw grid so we can apply our own header/footer handling.
    header=None prevents pandas from guessing headers.
    """
    if file_path.lower().endswith(".xlsx"):
        return pd.read_excel(file_path, header=None)
    return pd.read_csv(file_path, header=None, encoding="utf-8-sig")



# Backend-specific JSON loader that is resilient to LLM formatting issues
def _safe_json_loads(text: str) -> dict:
    if not text:
        return {}
    s = text.strip()

    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return {}

    return {}


def load_llm_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid LLM output (no JSON found):\n{text}")

    raw = match.group(0).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        try:
            return json.loads(raw.replace("\\", "\\\\"))
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from LLM:\n{raw}") from e