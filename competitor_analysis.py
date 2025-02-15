import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from serpapi import GoogleSearch
import os

# Download necessary NLTK data
nltk.download("vader_lexicon")


class CompetitorAnalysis:
    def __init__(self, api_key):
        self.api_key = api_key
        self.sia = SentimentIntensityAnalyzer()
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def search_competitors(self, query, num_results=5):
        params = {
            "engine": "google",
            "q": query + " competitors",
            "api_key": self.api_key,
            "num": num_results,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return [result["title"] for result in results.get("organic_results", [])]

    def scrape_website(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()

    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)

    def summarize_text(self, text, max_length=150):
        return self.summarizer(
            text, max_length=max_length, min_length=50, do_sample=False
        )[0]["summary_text"]

    def analyze_competitor(self, competitor_name):
        # Use SerpAPI to get the organic results
        params = {
            "engine": "google",
            "q": competitor_name,
            "api_key": self.api_key,
            "num": 1,  # Get just the first result
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        # Get the actual website URL from the first organic result
        organic_results = results.get("organic_results", [])
        if not organic_results:
            return {
                "name": competitor_name,
                "sentiment": {"neg": 0, "neu": 1, "pos": 0, "compound": 0},
                "summary": "No results found",
            }

        website_url = organic_results[0].get("link")
        try:
            # Scrape the actual website content
            response = requests.get(website_url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # Get only the visible text, removing scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            website_content = " ".join(soup.stripped_strings)

            # Limit content length to avoid overwhelming the summarizer
            website_content = website_content[:5000]  # Take first 5000 characters

            # Analyze sentiment
            sentiment = self.analyze_sentiment(website_content)

            # Summarize content
            summary = self.summarize_text(website_content, max_length=100)

        except Exception as e:
            return {
                "name": competitor_name,
                "sentiment": {"neg": 0, "neu": 1, "pos": 0, "compound": 0},
                "summary": f"Error accessing website: {str(e)}",
            }

        return {"name": competitor_name, "sentiment": sentiment, "summary": summary}

    def suggest_differentiation(self, competitors_data):
        # This is a simplified suggestion mechanism
        positive_aspects = []
        negative_aspects = []

        for competitor in competitors_data:
            if competitor["sentiment"]["compound"] > 0:
                positive_aspects.append(f"{competitor['name']} is viewed positively")
            else:
                negative_aspects.append(
                    f"{competitor['name']} has some negative aspects"
                )

        suggestion = (
            f"Consider addressing these pain points: {', '.join(negative_aspects)}. "
        )
        suggestion += (
            f"While also improving upon the positives: {', '.join(positive_aspects)}."
        )

        return suggestion


def main():
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("Please set the SERPAPI_API_KEY environment variable.")
        return

    analyzer = CompetitorAnalysis(api_key)

    # Example usage
    product_idea = "no-code website builder for designers"
    competitors = analyzer.search_competitors(product_idea)

    competitors_data = []
    for competitor in competitors:
        data = analyzer.analyze_competitor(competitor)
        competitors_data.append(data)
        print(f"Analyzed {competitor}:")
        print(f"Sentiment: {data['sentiment']}")
        print(f"Summary: {data['summary']}")
        print("---")

    suggestion = analyzer.suggest_differentiation(competitors_data)
    print("Differentiation suggestion:")
    print(suggestion)


if __name__ == "__main__":
    main()
