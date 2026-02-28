import pandas as pd
from google import genai
import io
import os
import re
import sys
import json
import numpy as np
import traceback
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

input_path = sys.argv[1]
output_path = sys.argv[2]
options = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {
    "missing": "drop",
    "outliers": "remove",
    "typos": "auto",
}

df = pd.read_csv(input_path)

buffer = io.StringIO()
df.info(buf=buffer)
info_string = buffer.getvalue()

head_string = df.head(10).to_string()
describe_string = df.describe(include='all').to_string()

sample_values = {}
for col in df.columns:
    sample_values[col] = [str(v) for v in df[col].dropna().unique()[:15]]
sample_string = "\n".join(f"  {col}: {vals}" for col, vals in sample_values.items())

if options["missing"] == "drop":
    missing_instruction = "Drop rows with any remaining missing/null values. Do not impute or fill."
else:
    missing_instruction = "Impute missing values: fill numeric columns with the median, and categorical/string columns with the mode (most frequent value)."

if options["outliers"] == "remove":
    outlier_instruction = "Remove extreme outliers that are clearly invalid based on the domain context of each column."
elif options["outliers"] == "cap":
    outlier_instruction = "Cap (winsorize) extreme outliers by clipping numeric values to the 1st and 99th percentile bounds instead of removing them."
else:
    outlier_instruction = "Leave outliers as-is. Do not remove or modify any outliers."

if options["typos"] == "auto":
    typo_instruction = "Fix typos and inconsistencies in categorical/string columns — correct misspellings, standardize formatting like replacing underscores with spaces, and unify equivalent values."
else:
    typo_instruction = "Leave categorical/string values as-is. Do not correct typos or standardize formatting."

prompt = f"""
You are an expert Data Scientist acting as an autonomous data-cleaning agent.
You must fix data quality issues based on the metadata and objectives below.

### DATA CONTEXT ###
1. Schema and Null Counts (df.info()):
{info_string}

2. Data Sample (df.head(10)):
{head_string}

3. Summary Statistics (df.describe(include='all')):
{describe_string}

4. Unique sample values per column:
{sample_string}

### YOUR OBJECTIVE ###
Write a Python script that performs ALL of the following:
1. Fix incorrect types — columns that should be numeric but contain invalid string entries should be converted using pd.to_numeric with errors='coerce'.
2. Handle mixed-type columns where some values are numeric and others are strings — standardize them to one consistent type.
3. Drop exact duplicate rows.
4. Drop any columns where more than 50% of the values are missing.
5. {missing_instruction}
6. {outlier_instruction}
7. {typo_instruction}

### STRICT CONSTRAINTS ###
- The data is already loaded in a variable called `df`. Do not include `pd.read_csv()`.
- Only output one clean block of Python code enclosed in ```python tags.
- No conversational text or explanations. Just the code.
- NEVER use chained assignment like `df[col].fillna(x, inplace=True)`. Use `df[col] = df[col].method()` instead. This is pandas with Copy-on-Write.
- When converting columns to numeric, use `pd.to_numeric(col, errors='coerce')`.
- The code must modify and return the `df` variable.
"""

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt
)

MAX_RETRIES = 3
conversation = [prompt]
original_df = df.copy()

for attempt in range(MAX_RETRIES):
    match = re.search(r'```python\n(.*?)\n```', response.text, re.DOTALL)
    if not match:
        break

    cleaning_code = match.group(1)
    local_scope = {'df': original_df.copy(), 'pd': pd, 'np': np}

    try:
        exec(cleaning_code, {}, local_scope)
        df = local_scope['df']
        df.to_csv(output_path, index=False)
        break
    except Exception as e:
        error_msg = traceback.format_exc()
        retry_prompt = f"""The code you generated raised an error:\n\n{error_msg}\n\nPlease fix the code and return a corrected version. Same constraints apply."""
        conversation.append(response.text)
        conversation.append(retry_prompt)

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents="\n".join(conversation)
        )
