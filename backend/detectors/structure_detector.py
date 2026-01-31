import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Set ROOT to the project root directory
sys.path.append(str(ROOT))

import os
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import re
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# Define a class for MERGED CELL & HEADER DETECTION
# ============================================================================

class SmartHeaderDetector:
    """
    Detects and handles merged cells, multi-row headers, title rows.
    """
    def __init__(self, file_path: str, sample_rows: int = 20):
        self.file_path = file_path
        self.sample_rows = sample_rows
        self.detected_header_row = None
        self.title_rows = []
        self.merged_cell_info = []

    def detect_structure(self) -> Dict[str, Any]:
        """
        Detect file structure including merged cells and title rows.
        """
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(self.sample_rows)]

        row_analysis = []
        for idx, line in enumerate(lines):
            if not line.strip():
                continue

            cells = [c.strip() for c in line.split(',')]
            filled_count = sum(1 for c in cells if c)

            row_analysis.append({
                'row_index': idx,
                'total_cells': len(cells),
                'filled_cells': filled_count,
                'fill_ratio': filled_count / len(cells) if cells else 0,
                'cells': cells,
                'is_mostly_empty': filled_count <= 2,
                'has_numeric': any(self._is_numeric(c) for c in cells if c)
            })

        # Title rows
        for analysis in row_analysis:
            if analysis['is_mostly_empty'] and not analysis['has_numeric']:
                self.title_rows.append(analysis['row_index'])

        # Header row
        for analysis in row_analysis:
            if (analysis['fill_ratio'] > 0.5 and
                analysis['row_index'] not in self.title_rows and
                not analysis['has_numeric']):
                self.detected_header_row = analysis['row_index']
                break

        # ✅ Only detect merged-cell patterns in the TOP area (titles + headers)
        # If header isn't found, fall back to only the first few rows.
        max_row = self.detected_header_row if self.detected_header_row is not None else 5
        self._detect_merged_cells(
            row_analysis=row_analysis,
            max_row=max_row,
            allowed_rows=set(self.title_rows + ([self.detected_header_row] if self.detected_header_row is not None else []))
        )

        return {
            'title_rows': self.title_rows,
            'header_row': self.detected_header_row,
            'skip_rows': self.title_rows,
            'merged_cells': self.merged_cell_info
        }

    def _is_numeric(self, value: str) -> bool:
        try:
            float(value.replace(',', '').replace('$', '').replace('€', ''))
            return True
        except:
            return False

    def _detect_merged_cells(self, row_analysis: List[Dict], max_row: int, allowed_rows: set[int] | None = None):
        """
        Detect merged cell patterns ONLY in the top rows.

        max_row: only consider rows with row_index <= max_row
        allowed_rows: if provided, only consider these exact rows (e.g., title + header rows)
        """
        for analysis in row_analysis:
            r = analysis['row_index']
            if r > max_row:
                continue
            if allowed_rows is not None and r not in allowed_rows:
                continue

            cells = analysis['cells']
            i = 0
            while i < len(cells):
                if cells[i]:
                    empty_count = 0
                    j = i + 1
                    while j < len(cells) and not cells[j]:
                        empty_count += 1
                        j += 1

                    if empty_count >= 2:
                        self.merged_cell_info.append({
                            'row': r,
                            'start_col': i,
                            'span': empty_count + 1,
                            'value': cells[i]
                        })
                    i = j
                else:
                    i += 1

    def read_cleaned_dataframe(self) -> pd.DataFrame:
        skip_rows = self.title_rows if self.title_rows else None
        try:
            df = pd.read_csv(self.file_path, skiprows=skip_rows)
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(self.file_path, skiprows=skip_rows, encoding=encoding)
                    df.columns = df.columns.str.strip()
                    return df
                except:
                    continue
            raise e
        
        
# ============================================================================

# Define a class for FOOTER DETECTION

# CSV Footer & Aggregate Row Detection System
# Detects and handles footer rows containing totals, subtotals, grand totals, etc.

# ============================================================================
def load_llm_json(text: str) -> Dict[str, Any]:
    if text is None:
        raise ValueError("LLM output is None")
    s = text.strip()
    if not s:
        raise ValueError("LLM output is empty")

    # Strip common code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # Try direct
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"footer_rows": obj}
    except json.JSONDecodeError:
        pass

    # Extract first JSON object or array anywhere in the text
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in LLM output. Head:\n{s[:400]}")

    raw = m.group(1).strip()
    obj = json.loads(raw)  # will raise if still invalid
    return obj if isinstance(obj, dict) else {"footer_rows": obj}

class FooterDetector:
    """
    LLM-only footer detection.
    Identifies final aggregate rows (totals / summaries) at the bottom of CSVs.
    """

    def __init__(self, file_path: str, llm):
        self.file_path = file_path
        self.llm = llm

    def detect_footer_rows(self, skip_rows=None, cell_i: int = 0) -> Dict[str, Any]:
        """
        Detect footer rows and also extract cell[cell_i] from each footer row,
        where cell_i refers to the i-th non-empty cell in that row.
        """
        skip_rows = sorted(set(skip_rows or []))

        try:
            df = pd.read_csv(self.file_path, skiprows=skip_rows)
        except Exception as e:
            return {"success": False, "error": str(e)}

        total_rows = len(df)
        if total_rows == 0:
            return {
                "success": True,
                "footer_rows": [],
                "data_end_row": 0,
                "total_rows": 0,
                "footer_count": 0,
                "footer_cells_i": [],
            }

        llm_rows = self._detect_by_llm(df)

        # 🚨 Only trust footer rows near the bottom
        bottom_cutoff = max(0, total_rows - 10)
        footer_rows = sorted([r for r in llm_rows if r >= bottom_cutoff])

        data_end_row = min(footer_rows) if footer_rows else total_rows

        # ✅ Extract cell[i] from each footer row (i-th non-empty cell)
        footer_cells_i = []
        for r in footer_rows:
            try:
                row = df.iloc[r]

                # Keep only non-null & non-blank cells (preserve original values)
                mask = row.notna() & row.astype(str).str.strip().ne("")
                non_empty = row[mask]

                if 0 <= cell_i < len(non_empty):
                    footer_cells_i.append(non_empty.iloc[cell_i])
                else:
                    footer_cells_i.append(None)
            except Exception:
                footer_cells_i.append(None)

        return {
            "success": True,
            "footer_rows": footer_rows,       # pandas df indices
            "data_end_row": data_end_row,
            "total_rows": total_rows,
            "footer_count": len(footer_rows),
            "footer_cells_i": footer_cells_i, # ✅ aligned with footer_rows
        }

    def _detect_by_llm(self, df: pd.DataFrame) -> list[int]:
        tail_n = min(20, len(df))
        df_tail = df.tail(tail_n)

        rows_info = [
            {
                "row_index": int(idx),
                "values": [str(v) if pd.notna(v) else "" for v in row.tolist()]
            }
            for idx, row in df_tail.iterrows()
        ]

        system_prompt = (
            "Return ONLY valid JSON.\n"
            "Schema: {\"footer_rows\": [int, ...]}\n"
            "Footer rows are final aggregate rows such as Total, Subtotal, "
            "Grand Total, Sum, Average, Count, GST/Tax, Net/Gross.\n"
            "Ignore section totals that appear mid-table."
        )

        user_prompt = json.dumps(rows_info, ensure_ascii=False)

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            parsed = load_llm_json(response.content)
            return [int(i) for i in parsed.get("footer_rows", [])]
        except Exception as e:
            print(f"LLM footer detection failed: {e}")
            return []

# ============================================================================
# ENHANCED SMART HEADER DETECTOR (WITH FOOTER DETECTION)
# ============================================================================

class SmartStructureDetector:
    """
    Enhanced detector that finds both headers AND footers.
    """
    
    def __init__(self, file_path: str, llm: Optional[ChatOpenAI] = None):
        self.file_path = file_path
        self.llm = llm or ChatOpenAI(model="gpt-5-nano", temperature=0)
    
    def detect_complete_structure(self) -> Dict[str, Any]:
        """
        Detect complete file structure:
        - Title rows (top)
        - Header row
        - Data rows
        - Footer rows (bottom)
        """
        # First detect header structure (title rows, headers)
        header_detector = SmartHeaderDetector(self.file_path)
        header_info = header_detector.detect_structure()
        
        # Then detect footer structure
        footer_detector = FooterDetector(self.file_path, self.llm)
        footer_info = footer_detector.detect_footer_rows()
        
        table_spec = {
            "table_region": {
                "header_rows_idx": header_info.get('title_rows', []),
                "data_start_row_idx": header_info.get('header_row'),
                "data_end_row_idx": footer_info.get('data_end_row'),
                "footer_rows_idx": footer_info.get('footer_rows', [])
            },
            "header_hints": {
                "header_label_row": header_info.get('merged_cells', []),
            },
            "footer_hints": {
                "footer_label_row": footer_info.get('footer_cells_i', [])
            },
            
        }
        
        return table_spec

test_file = "C:\\Users\\sovan\\Downloads\\Tester 1.csv"

structure_detector = SmartStructureDetector(test_file)
structure_info = structure_detector.detect_complete_structure()

footer_detector = FooterDetector(test_file, llm=ChatOpenAI(model="gpt-5-nano", temperature=0))
footer_info = footer_detector.detect_footer_rows()


header_detector = SmartHeaderDetector(test_file)
header_info = header_detector.detect_structure()