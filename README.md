# AnalyzeAI

End-to-end data analysis workflow that cleans messy CSVs and easily do statistical modeling and inference.

## Features

### Step 1: Clean the Data
Upload any CSV and the agent will automatically:
- **Fix incorrect types** - converts columns that should be numeric but contain invalid entries
- **Handle mixed-type columns** - standardizes columns with both numbers and strings
- **Drop duplicates** - removes exact duplicate rows
- **Drop high-null columns** - removes columns where >50% of values are missing
- **Fix spelling errors** — corrects typos and standardizes inconsistent categorical values

### User-Configurable Options
| Option | Choices |
|---|---|
| **Missing Values** | Drop rows · Impute (median for numeric, mode for categorical) |
| **Outliers** | Remove · Cap to 1st/99th percentile · Keep |
| **Typo Cleanup** | Auto-correct misspellings & standardize categories · Skip |
