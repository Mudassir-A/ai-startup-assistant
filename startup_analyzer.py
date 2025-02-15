import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from serpapi.google_search import GoogleSearch
import pandas as pd
from typing import List, Dict, Any
import asyncio
import aiohttp
from datetime import datetime
import json
import os
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StartupAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the StartupAnalyzer with necessary API keys and models."""
        self.api_key = api_key
        self.sia = SentimentIntensityAnalyzer()
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.classifier = pipeline("zero-shot-classification")

        # Download required NLTK data
        try:
            nltk.data.find("vader_lexicon")
        except LookupError:
            nltk.download("vader_lexicon")

        # Categories for market analysis
        self.market_categories = [
            "B2B",
            "B2C",
            "SaaS",
            "E-commerce",
            "Mobile App",
            "Enterprise",
            "Small Business",
            "Consumer",
        ]

        # Add new model for business viability analysis
        self.business_classifier = pipeline(
            "text-classification", model="ProsusAI/finbert"
        )

        # Add viability assessment categories
        self.viability_factors = [
            "market demand",
            "competition level",
            "innovation potential",
            "regulatory environment",
            "scalability",
            "revenue potential",
        ]

    async def analyze_market(self, business_idea: str) -> Dict[str, Any]:
        """
        Comprehensive market analysis for a business idea.
        """
        try:
            # Parallel execution of different analyses
            competitors_data = await self.get_competitor_analysis(business_idea)
            market_size = await self.estimate_market_size(business_idea)
            trends = await self.analyze_market_trends(business_idea)
            target_segments = await self.identify_target_segments(business_idea)

            # Add viability analysis
            viability_analysis = await self.analyze_business_viability(
                business_idea, competitors_data, market_size, trends, target_segments
            )

            return {
                "timestamp": datetime.now().isoformat(),
                "business_idea": business_idea,
                "market_analysis": {
                    "competitors": competitors_data,
                    "market_size": market_size,
                    "trends": trends,
                    "target_segments": target_segments,
                    "recommendations": self.generate_recommendations(
                        competitors_data, trends
                    ),
                    "viability_analysis": viability_analysis,
                },
            }
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            raise

    async def get_competitor_analysis(self, business_idea: str) -> List[Dict[str, Any]]:
        """
        Enhanced competitor analysis with more detailed insights.
        """
        competitors = await self.search_competitors(business_idea)
        async with aiohttp.ClientSession() as session:
            tasks = [self.analyze_competitor(session, comp) for comp in competitors]
            results = await asyncio.gather(*tasks)

        return self.enrich_competitor_data(results)

    async def analyze_competitor(
        self, session: aiohttp.ClientSession, competitor_name: str
    ) -> Dict[str, Any]:
        """
        Enhanced competitor analysis with better error handling and timeouts.
        """
        try:
            # Search for competitor info using a more specific query
            params = {
                "engine": "google",
                "q": f"{competitor_name} company information revenue employees",
                "api_key": self.api_key,
                "num": 1,
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            if not results.get("organic_results"):
                return self.create_empty_competitor_analysis(competitor_name)

            website_url = results["organic_results"][0]["link"]

            # Add headers to mimic a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            try:
                async with session.get(
                    website_url, headers=headers, timeout=15
                ) as response:
                    if response.status != 200:
                        # Try to get snippet from Google results if website access fails
                        return self.analyze_from_search_snippet(
                            competitor_name, results["organic_results"][0]
                        )

                    content = await response.text()
                    analysis = await self.analyze_website_content(
                        content, competitor_name
                    )
                    analysis["website"] = website_url
                    return analysis
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    f"Could not access website for {competitor_name}, using search snippet instead"
                )
                return self.analyze_from_search_snippet(
                    competitor_name, results["organic_results"][0]
                )

        except Exception as e:
            logger.error(f"Error analyzing competitor {competitor_name}: {str(e)}")
            return self.create_empty_competitor_analysis(competitor_name)

    def analyze_from_search_snippet(
        self, competitor_name: str, search_result: Dict
    ) -> Dict[str, Any]:
        """
        Analyze competitor based on Google search snippet when website is inaccessible.
        """
        snippet = search_result.get("snippet", "")
        title = search_result.get("title", "")

        # Combine title and snippet for analysis
        text_content = f"{title}. {snippet}"

        # Perform sentiment analysis on the snippet
        sentiment = self.analyze_sentiment(text_content)

        # Classify business based on snippet
        business_category = self.classify_business(text_content)

        return {
            "name": competitor_name,
            "sentiment": sentiment,
            "business_category": business_category,
            "summary": snippet[:200] if snippet else "No summary available",
            "website": search_result.get("link"),
            "analysis_timestamp": datetime.now().isoformat(),
            "data_source": "search_snippet",
            "market_position": {"strength": "unknown", "confidence_score": 0.5},
        }

    async def search_competitors(self, query: str, num_results: int = 10) -> List[str]:
        """
        Enhanced competitor search with better query formatting and filtering.
        """
        # Create a more focused search query
        search_query = f"{query} top companies competitors startups market leaders"

        params = {
            "engine": "google",
            "q": search_query,
            "api_key": self.api_key,
            "num": num_results + 5,  # Request extra results for filtering
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            # Filter and clean results
            competitors = []
            for result in results.get("organic_results", []):
                title = result.get("title", "").strip()
                # Skip results that are likely not relevant
                if any(
                    x in title.lower()
                    for x in ["how to", "guide", "tutorial", "list of"]
                ):
                    continue
                if title:
                    competitors.append(title)
                if len(competitors) >= num_results:
                    break

            return competitors[:num_results]

        except Exception as e:
            logger.error(f"Error in competitor search: {str(e)}")
            return []

    async def analyze_website_content(
        self, content: str, competitor_name: str
    ) -> Dict[str, Any]:
        """
        Analyze website content with enhanced metrics.
        """
        soup = BeautifulSoup(content, "html.parser")
        text_content = self.clean_text_content(soup)

        # Parallel processing of different analyses
        with ThreadPoolExecutor() as executor:
            sentiment_future = executor.submit(self.analyze_sentiment, text_content)
            category_future = executor.submit(self.classify_business, text_content)
            summary_future = executor.submit(self.summarize_text, text_content)

        return {
            "name": competitor_name,
            "sentiment": sentiment_future.result(),
            "business_category": category_future.result(),
            "summary": summary_future.result(),
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def classify_business(self, text: str) -> Dict[str, float]:
        """
        Classify business type using zero-shot classification.
        """
        result = self.classifier(text, self.market_categories)
        return dict(zip(result["labels"], result["scores"]))

    async def estimate_market_size(self, business_idea: str) -> Dict[str, Any]:
        """
        Estimate market size using available data.
        """
        # Implementation would depend on available market size data sources
        # This is a placeholder that you would need to implement based on your data sources
        return {
            "estimated_size": "Placeholder for market size estimation",
            "growth_rate": "Placeholder for growth rate",
            "confidence_level": "medium",
        }

    async def analyze_market_trends(self, business_idea: str) -> List[Dict[str, Any]]:
        """
        Analyze current market trends.
        """
        # Implementation would depend on your trend data sources
        # This is a placeholder that you would need to implement
        return [
            {
                "trend": "Placeholder trend 1",
                "impact": "high",
                "description": "Trend description",
            }
        ]

    def generate_recommendations(
        self, competitors_data: List[Dict], trends: List[Dict]
    ) -> List[str]:
        """
        Generate strategic recommendations based on analysis.
        """
        recommendations = []

        # Analyze competitor strengths and weaknesses
        strengths = []
        weaknesses = []
        for comp in competitors_data:
            if comp["sentiment"]["compound"] > 0.2:
                strengths.append(f"Strong online presence like {comp['name']}")
            elif comp["sentiment"]["compound"] < -0.2:
                weaknesses.append(f"Improve upon {comp['name']}'s weaknesses")

        # Generate strategic recommendations
        if strengths:
            recommendations.append(f"Consider: {', '.join(strengths)}")
        if weaknesses:
            recommendations.append(f"Opportunity areas: {', '.join(weaknesses)}")

        return recommendations

    @staticmethod
    def clean_text_content(soup: BeautifulSoup) -> str:
        """
        Clean and prepare website content for analysis.
        """
        for script in soup(["script", "style", "meta", "noscript"]):
            script.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:10000]  # Limit content length

    @staticmethod
    def create_empty_competitor_analysis(competitor_name: str) -> Dict[str, Any]:
        """
        Create empty analysis structure for failed competitor analysis.
        """
        return {
            "name": competitor_name,
            "sentiment": {"neg": 0, "neu": 1, "pos": 0, "compound": 0},
            "business_category": {},
            "summary": "Analysis failed",
            "website": None,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def save_analysis(self, analysis_data: Dict[str, Any], filename: str = None) -> str:
        """
        Save analysis results to a file.
        """
        if filename is None:
            filename = (
                f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        with open(filename, "w") as f:
            json.dump(analysis_data, f, indent=2)

        return filename

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using NLTK's VADER sentiment analyzer.
        """
        return self.sia.polarity_scores(text)

    def summarize_text(self, text: str, max_length: int = 130) -> str:
        """
        Summarize text using the BART model.
        """
        if len(text.split()) < 30:  # If text is too short, return as is
            return text
        try:
            summary = self.summarizer(
                text, max_length=max_length, min_length=30, do_sample=False
            )
            return summary[0]["summary_text"]
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return text[:max_length] + "..."

    def enrich_competitor_data(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich competitor analysis data with additional insights.
        """
        enriched_data = []
        for result in results:
            if result and isinstance(result, dict):
                # Add market position assessment
                sentiment = result.get("sentiment", {})
                compound_score = sentiment.get("compound", 0)

                result["market_position"] = {
                    "strength": (
                        "strong"
                        if compound_score > 0.2
                        else "weak" if compound_score < -0.2 else "neutral"
                    ),
                    "confidence_score": abs(compound_score),
                }

                # Add analysis timestamp if not present
                if "analysis_timestamp" not in result:
                    result["analysis_timestamp"] = datetime.now().isoformat()

                enriched_data.append(result)

        return enriched_data

    async def identify_target_segments(
        self, business_idea: str
    ) -> List[Dict[str, Any]]:
        """
        Identify potential target market segments.
        """
        # This is a placeholder implementation
        segments = [
            {
                "segment": "Early Adopters",
                "characteristics": "Tech-savvy individuals interested in fitness innovation",
                "potential": "High",
            },
            {
                "segment": "Fitness Enthusiasts",
                "characteristics": "Regular gym-goers looking for personalized guidance",
                "potential": "Medium",
            },
            {
                "segment": "Busy Professionals",
                "characteristics": "Time-constrained individuals seeking efficient workout solutions",
                "potential": "High",
            },
        ]
        return segments

    async def analyze_business_viability(
        self,
        business_idea: str,
        competitors_data: List[Dict],
        market_size: Dict,
        trends: List[Dict],
        target_segments: List[Dict],
    ) -> Dict[str, Any]:
        """
        Analyze the viability of the business idea based on market data.
        """
        try:
            # Prepare analysis text combining all data
            analysis_text = f"""
            Business Idea: {business_idea}
            Market Size: {market_size.get('estimated_size', 'Unknown')}
            Growth Rate: {market_size.get('growth_rate', 'Unknown')}
            Number of Competitors: {len(competitors_data)}
            Target Segments: {len(target_segments)}
            """

            # Analyze sentiment and market potential
            sentiment_scores = []
            for comp in competitors_data:
                if comp.get("sentiment"):
                    sentiment_scores.append(comp["sentiment"]["compound"])

            # Calculate market saturation
            market_saturation = len(competitors_data) / 10  # Normalize to 0-1 scale

            # Analyze using FinBERT
            financial_sentiment = self.business_classifier(analysis_text)

            # Calculate viability scores for each factor
            viability_scores = {
                "market_demand": self._calculate_market_demand_score(
                    target_segments, trends
                ),
                "competition_level": max(0, 1 - market_saturation),
                "innovation_potential": self._analyze_innovation_potential(
                    business_idea, competitors_data
                ),
                "scalability": self._analyze_scalability(target_segments),
                "revenue_potential": self._calculate_revenue_potential(
                    market_size, trends
                ),
            }

            # Calculate overall viability score (0-100)
            overall_score = sum(viability_scores.values()) / len(viability_scores) * 100

            # Generate detailed reasoning
            reasoning = self._generate_viability_reasoning(
                viability_scores, financial_sentiment[0], trends, competitors_data
            )

            return {
                "overall_viability_score": round(overall_score, 2),
                "confidence_level": self._calculate_confidence_level(viability_scores),
                "factor_scores": viability_scores,
                "reasoning": reasoning,
                "recommendation": self._generate_viability_recommendation(
                    overall_score
                ),
            }

        except Exception as e:
            logger.error(f"Error in viability analysis: {str(e)}")
            return {
                "overall_viability_score": 0,
                "confidence_level": "low",
                "factor_scores": {},
                "reasoning": f"Analysis failed: {str(e)}",
                "recommendation": "Unable to make a recommendation due to analysis failure",
            }

    def _calculate_market_demand_score(
        self, target_segments: List[Dict], trends: List[Dict]
    ) -> float:
        """Calculate market demand score based on segments and trends."""
        segment_score = min(len(target_segments) / 5, 1.0)  # Normalize to 0-1
        trend_score = sum(1 for trend in trends if trend.get("impact") == "high") / len(
            trends
        )
        return (segment_score + trend_score) / 2

    def _analyze_innovation_potential(
        self, business_idea: str, competitors_data: List[Dict]
    ) -> float:
        """Analyze innovation potential compared to competitors."""
        # Compare business idea against competitor summaries
        innovation_scores = []
        for comp in competitors_data:
            if comp.get("summary"):
                similarity = self.classifier(
                    business_idea, [comp["summary"]], multi_label=False
                )
                innovation_scores.append(1 - similarity["scores"][0])

        return (
            sum(innovation_scores) / len(innovation_scores)
            if innovation_scores
            else 0.5
        )

    def _analyze_scalability(self, target_segments: List[Dict]) -> float:
        """Analyze scalability based on target segments."""
        high_potential_segments = sum(
            1
            for segment in target_segments
            if segment.get("potential", "").lower() == "high"
        )
        return min(high_potential_segments / len(target_segments), 1.0)

    def _calculate_revenue_potential(
        self, market_size: Dict, trends: List[Dict]
    ) -> float:
        """Calculate revenue potential based on market size and trends."""
        # Simplified scoring based on growth rate and market size
        growth_score = 0.7  # Default moderate score
        if "growth_rate" in market_size:
            growth_rate = market_size["growth_rate"]
            if isinstance(growth_rate, (int, float)):
                growth_score = min(growth_rate / 20, 1.0)  # Normalize to 0-1

        positive_trends = sum(
            1 for trend in trends if trend.get("impact", "").lower() == "high"
        )
        trend_score = min(positive_trends / len(trends), 1.0)

        return (growth_score + trend_score) / 2

    def _calculate_confidence_level(self, scores: Dict[str, float]) -> str:
        """Calculate confidence level of the analysis."""
        avg_score = sum(scores.values()) / len(scores)
        if avg_score > 0.8:
            return "high"
        elif avg_score > 0.5:
            return "medium"
        return "low"

    def _generate_viability_reasoning(
        self,
        scores: Dict[str, float],
        financial_sentiment: Dict,
        trends: List[Dict],
        competitors_data: List[Dict],
    ) -> List[str]:
        """Generate detailed reasoning for the viability analysis."""
        reasoning = []

        # Add reasoning based on scores
        for factor, score in scores.items():
            if score > 0.7:
                reasoning.append(f"Strong {factor.replace('_', ' ')} potential")
            elif score < 0.3:
                reasoning.append(f"Concerning {factor.replace('_', ' ')} indicators")

        # Add trend-based reasoning
        positive_trends = [t for t in trends if t.get("impact") == "high"]
        if positive_trends:
            reasoning.append(
                f"Favorable market trends: {len(positive_trends)} high-impact trends identified"
            )

        # Add competition-based reasoning
        strong_competitors = sum(
            1
            for c in competitors_data
            if c.get("market_position", {}).get("strength") == "strong"
        )
        if strong_competitors > len(competitors_data) / 2:
            reasoning.append("High competition from established players")

        return reasoning

    def _generate_viability_recommendation(self, overall_score: float) -> str:
        """Generate final recommendation based on overall viability score."""
        if overall_score >= 75:
            return "Highly recommended to proceed with the business idea"
        elif overall_score >= 60:
            return "Proceed with caution, address identified weaknesses"
        elif overall_score >= 40:
            return "Significant modifications needed before proceeding"
        else:
            return "Not recommended to proceed in current form"


async def main():
    """
    Example usage of the StartupAnalyzer.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment variable
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY not found in .env file. Please add it.")

    # Initialize analyzer with API key
    analyzer = StartupAnalyzer(api_key)

    # Example business idea
    business_idea = "AI-powered personal fitness coach mobile app"

    try:
        # Perform analysis
        analysis_results = await analyzer.analyze_market(business_idea)

        # Save results
        filename = analyzer.save_analysis(analysis_results)
        logger.info(f"Analysis saved to {filename}")

        # Print summary
        print("\n=== Market Analysis Summary ===")
        print(f"Business Idea: {business_idea}")
        print("\nKey Competitors:")
        for comp in analysis_results["market_analysis"]["competitors"]:
            print(f"- {comp['name']}: {comp['summary']}")

        print("\nRecommendations:")
        for rec in analysis_results["market_analysis"]["recommendations"]:
            print(f"- {rec}")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
