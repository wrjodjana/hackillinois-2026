import pandas as pd
from google import genai
import io
import os
import re
import sys
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

input_path = sys.argv[1]
output_path = sys.argv[2]

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

prompt = f"""
You are an expert Data Scientist acting as an autonomous data-cleaning agent.
You must automatically fix ALL data quality issues based on the metadata below.

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
1. Drop rows with missing/null values.
2. Fix incorrect types — columns that should be numeric but contain invalid string entries should be converted using pd.to_numeric with errors='coerce' before dropping.
3. Fix typos and inconsistencies in categorical/string columns — correct misspellings, standardize formatting like replacing underscores with spaces, and unify equivalent values.
4. Remove extreme outliers that are clearly invalid based on the domain context of each column.
5. Handle mixed-type columns where some values are numeric and others are strings — standardize them to one consistent type.

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

import numpy as np
import traceback

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




