import pandas as pd
from google import genai
import io
import os
import re
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

df = pd.read_csv('smartwatch.csv')
df

description = df.describe()
info = df.info()
head = df.head(3)

buffer = io.StringIO()
df.info(buf=buffer)
info_string = buffer.getvalue()

prompt = f"""
You are an expert Data Scientist acting as an autonomous data-cleaning agent.
You must automatically fix issues in the dataset based strictly on the metadata below.

### DATA CONTEXT ###
1. Data Sample (df.head(3)):
{head}

2. Schema and Null Counts (df.info()):
{info}

3. Summary Statistics (df.describe()):
{description}

### YOUR OBJECTIVE ###
Write a Python script that automatically handles missing values, incorrect types, and extreme outliers.

### STRICT CONSTRAINTS ###
- Assume the data is already loaded into a variable named `df`.
- Do not include `pd.read_csv()` in your code.
- Only output one clean block of Python code enclosed in ```python tags.
- Do not provide any conversational text or explanations. Just the code.
- For missing values, drop the rows containing them. Do not impute or fill missing values.
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
        df.to_csv('cleaned_smartwatch.csv', index=False)
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




