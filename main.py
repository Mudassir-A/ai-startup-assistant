import asyncio
import os
from dotenv import load_dotenv
from startup_analyzer import StartupAnalyzer

# Load environment variables from .env file
load_dotenv()

# Get API key
api_key = os.getenv("SERPAPI_API_KEY")
if not api_key:
    raise ValueError("SERPAPI_API_KEY not found in .env file. Please add it.")

# Initialize the analyzer with the API key
analyzer = StartupAnalyzer(api_key)

# Now you can use your analyzer
async def analyze_my_idea():
    business_idea = "AI-powered personal fitness coach mobile app"
    results = await analyzer.analyze_market(business_idea)
    print(results)

if __name__ == "__main__":
    asyncio.run(analyze_my_idea())