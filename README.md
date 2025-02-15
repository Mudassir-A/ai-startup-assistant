# Startup Analysis Tools

## What it does

- **networking.py**: Finds and analyzes potential mentors in your field using LinkedIn profiles and Google search.
- **startup_analyzer.py**: Analyzes market viability and competitors for your business idea using web data.

## Setup & Running

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in project root with:
```
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
SERPAPI_API_KEY=your_serpapi_key
```

3. Run the tools:
```bash
# To find mentors
python networking.py

# To analyze your startup idea
python startup_analyzer.py
```

## Getting API Keys

### networking.py requires:
- Get Google API Key: https://console.cloud.google.com/
- Get Custom Search ID: https://programmablesearchengine.google.com/

### startup_analyzer.py requires:
- Get SerpAPI Key: https://serpapi.com/