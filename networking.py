import requests
import pandas as pd
from typing import List, Dict, Any
from transformers import pipeline
from googleapiclient.discovery import build
import asyncio
import aiohttp
from datetime import datetime
import logging
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv


class MentorFinder:
    def __init__(self):
        """Initialize the MentorFinder with necessary models and API keys."""
        # Load environment variables
        load_dotenv()  # This will load .env file if it exists

        # Get API keys from environment
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

        # Validate API keys
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please add it to your .env file or environment variables."
            )
        if not self.google_cse_id:
            raise ValueError(
                "GOOGLE_CSE_ID not found in environment variables. "
                "Please add it to your .env file or environment variables."
            )

        # Initialize models
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("MentorFinder")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    async def find_potential_mentors(
        self, field: str, location: str = None, min_experience: int = 5
    ) -> List[Dict[str, Any]]:
        """Find potential mentors based on field and criteria."""
        try:
            # Create search queries
            queries = [
                f"{field} entrepreneur founder linkedin",
                f"{field} startup CEO linkedin profile",
                f"{field} business mentor linkedin",
                f"{field} industry expert linkedin",
            ]

            if location:
                queries = [f"{q} {location}" for q in queries]

            all_mentors = []
            async with aiohttp.ClientSession() as session:
                for query in queries:
                    mentors = await self._search_mentors(session, query)
                    all_mentors.extend(mentors)

            # Remove duplicates and filter
            unique_mentors = self._remove_duplicates(all_mentors)
            filtered_mentors = self._filter_mentors(unique_mentors, min_experience)

            # Score and rank mentors
            ranked_mentors = self._rank_mentors(filtered_mentors, field)

            return ranked_mentors[:10]  # Return top 10 matches

        except Exception as e:
            self.logger.error(f"Error finding mentors: {str(e)}")
            return []

    async def _search_mentors(
        self, session: aiohttp.ClientSession, query: str
    ) -> List[Dict[str, Any]]:
        """Search for potential mentors using Google Custom Search API."""
        try:
            service = build("customsearch", "v1", developerKey=self.google_api_key)

            self.logger.info(f"Searching for: {query}")

            try:
                # Execute the search
                result = (
                    service.cse().list(q=query, cx=self.google_cse_id, num=10).execute()
                )

                mentors = []
                for item in result.get("items", []):
                    self.logger.info(f"Processing result: {item.get('title', '')}")
                    mentor_info = await self._extract_mentor_info(session, item)
                    if mentor_info:
                        mentors.append(mentor_info)
                        self.logger.info(f"Added mentor: {mentor_info['name']}")

                return mentors

            except Exception as api_error:
                if "403" in str(api_error):
                    self.logger.error(
                        "Google Custom Search API is not properly enabled."
                    )
                    print("\nTo fix this error:")
                    print("1. Go to https://console.cloud.google.com/")
                    print("2. Select your project")
                    print("3. Go to 'APIs & Services' > 'Library'")
                    print("4. Search for 'Custom Search API'")
                    print("5. Click 'Enable'")
                    print("6. Go to 'APIs & Services' > 'Credentials'")
                    print("7. Create or use an existing API key")
                    print("8. Go to https://programmablesearchengine.google.com/")
                    print("9. Create a new search engine")
                    print("10. Get your Search Engine ID (cx)")
                    print("\nThen update your environment variables with the new keys.")
                    return []
                raise

        except Exception as e:
            self.logger.error(f"Error in mentor search: {str(e)}")
            return []

    async def _extract_mentor_info(
        self, session: aiohttp.ClientSession, result: Dict
    ) -> Dict[str, Any]:
        """Extract mentor information from search result."""
        try:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")

            # Skip if not a LinkedIn profile
            if "linkedin.com/in/" not in link.lower():
                return None

            # Skip if not likely to be about a person
            if not self._is_likely_person(title, snippet):
                return None

            # Extract name and basic info
            name = self._extract_name(title)
            if not name:
                return None

            # Try to get more info from LinkedIn profile
            profile_data = await self._scrape_linkedin_preview(session, link)

            # Combine all information
            profile_info = {
                "name": name,
                "title": profile_data.get("title", title),
                "summary": profile_data.get("summary", snippet),
                "profile_url": link,
                "expertise": self._extract_expertise(
                    profile_data.get("summary", snippet)
                ),
                "experience_years": profile_data.get(
                    "experience_years", self._extract_experience(snippet)
                ),
                "contact_info": {"linkedin": link},
                "sentiment_score": self._analyze_sentiment(snippet),
                "source": "linkedin",
                "last_updated": datetime.now().isoformat(),
            }

            return profile_info

        except Exception as e:
            self.logger.error(f"Error extracting mentor info: {str(e)}")
            return None

    async def _scrape_linkedin_preview(
        self, session: aiohttp.ClientSession, profile_url: str
    ) -> Dict[str, Any]:
        """Safely scrape public LinkedIn profile preview."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            async with session.get(profile_url, headers=headers) as response:
                if response.status != 200:
                    return {}

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Extract available information
                data = {"title": "", "summary": "", "experience_years": 0}

                # Try to find title/headline
                title_elem = soup.find("title")
                if title_elem:
                    data["title"] = title_elem.text.split(" | ")[0]

                # Try to find summary/about
                summary_elem = soup.find("section", {"id": "about"})
                if summary_elem:
                    data["summary"] = summary_elem.text.strip()

                # Try to find experience
                exp_section = soup.find("section", {"id": "experience"})
                if exp_section:
                    # Calculate total years of experience
                    years = set()
                    year_pattern = r"\b20\d{2}\b"
                    found_years = re.findall(year_pattern, exp_section.text)
                    if found_years:
                        years.update(map(int, found_years))
                        if len(years) >= 2:
                            data["experience_years"] = max(years) - min(years)

                return data

        except Exception as e:
            self.logger.error(f"Error scraping LinkedIn preview: {str(e)}")
            return {}

    def _is_likely_person(self, title: str, snippet: str) -> bool:
        """
        Check if the search result is likely about a person.
        """
        lowercase_text = f"{title} {snippet}".lower()

        # Keywords that suggest it's about a person
        person_indicators = [
            "founder",
            "ceo",
            "entrepreneur",
            "professional",
            "expert",
            "specialist",
            "mentor",
            "advisor",
        ]

        # Keywords that suggest it's not about a person
        non_person_indicators = [
            "company",
            "corporation",
            "ltd",
            "llc",
            "inc",
            "website",
            "platform",
            "service",
            "product",
        ]

        person_score = sum(1 for word in person_indicators if word in lowercase_text)
        non_person_score = sum(
            1 for word in non_person_indicators if word in lowercase_text
        )

        return person_score > non_person_score

    def _extract_name(self, title: str) -> str:
        """
        Extract person's name from title.
        """
        # Remove common suffixes and prefixes
        suffixes = [
            "CEO",
            "Founder",
            "Co-Founder",
            "Expert",
            "Mentor",
            "Advisor",
            "Professional",
            "Specialist",
        ]

        name = title
        for suffix in suffixes:
            name = name.split(f" {suffix}")[0]
            name = name.split(f" - ")[0]

        return name.strip()

    def _extract_expertise(self, text: str) -> List[str]:
        """
        Extract areas of expertise from text.
        """
        # Common expertise keywords
        expertise_keywords = [
            "specialist in",
            "expert in",
            "experienced in",
            "focused on",
            "specializing in",
            "expertise in",
        ]

        expertise = []
        text_lower = text.lower()

        for keyword in expertise_keywords:
            if keyword in text_lower:
                start_idx = text_lower.find(keyword) + len(keyword)
                end_idx = text_lower.find(".", start_idx)
                if end_idx != -1:
                    expertise_text = text[start_idx:end_idx].strip()
                    expertise.append(expertise_text)

        return list(set(expertise))

    def _extract_experience(self, text: str) -> int:
        """
        Extract years of experience from text.
        """
        try:
            # Look for patterns like "X years of experience"
            text_lower = text.lower()
            if "years" in text_lower and "experience" in text_lower:
                words = text_lower.split()
                for i, word in enumerate(words):
                    if word == "years" and i > 0:
                        try:
                            return int(words[i - 1])
                        except ValueError:
                            continue
            return 0
        except Exception:
            return 0

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of the text description.
        """
        try:
            result = self.sentiment_analyzer(text[:512])[0]
            score = (
                result["score"] if result["label"] == "POSITIVE" else -result["score"]
            )
            return score
        except Exception:
            return 0.0

    def _remove_duplicates(self, mentors: List[Dict]) -> List[Dict]:
        """
        Remove duplicate mentor entries based on name and URL.
        """
        seen = set()
        unique_mentors = []

        for mentor in mentors:
            key = (mentor["name"], mentor.get("profile_url", ""))
            if key not in seen:
                seen.add(key)
                unique_mentors.append(mentor)

        return unique_mentors

    def _filter_mentors(self, mentors: List[Dict], min_experience: int) -> List[Dict]:
        """
        Filter mentors based on criteria.
        """
        filtered = []
        for mentor in mentors:
            if (
                mentor.get("experience_years", 0) >= min_experience
                and mentor.get("expertise")
                and mentor.get("sentiment_score", 0) > 0
            ):
                filtered.append(mentor)
        return filtered

    def _rank_mentors(self, mentors: List[Dict], field: str) -> List[Dict]:
        """
        Rank mentors based on relevance to the field.
        """
        if not mentors:
            return []

        # Encode field and mentor descriptions
        field_embedding = self.model.encode([field])[0]

        for mentor in mentors:
            # Create mentor description
            mentor_text = f"{mentor['expertise']} {mentor['summary']}"
            mentor_embedding = self.model.encode([mentor_text])[0]

            # Calculate similarity score
            similarity = cosine_similarity(
                field_embedding.reshape(1, -1), mentor_embedding.reshape(1, -1)
            )[0][0]

            # Calculate final score
            mentor["relevance_score"] = similarity * (
                0.6  # Base relevance
                + 0.2 * min(mentor.get("experience_years", 0) / 10, 1)  # Experience
                + 0.2 * max(mentor.get("sentiment_score", 0), 0)  # Sentiment
            )

        # Sort by relevance score
        ranked_mentors = sorted(
            mentors, key=lambda x: x.get("relevance_score", 0), reverse=True
        )

        return ranked_mentors


async def main():
    try:
        # Initialize MentorFinder
        finder = MentorFinder()

        # Get search parameters
        print("\nMentor Finder - Find industry experts and potential mentors")
        print("--------------------------------------------------------")
        field = input(
            "\nEnter your field of interest (e.g., AI, blockchain, biotech): "
        )
        location = input("Enter preferred location (or press Enter to skip): ")

        while True:
            try:
                min_experience = int(
                    input("Enter minimum years of experience required (1-40): ")
                )
                if 1 <= min_experience <= 40:
                    break
                print("Please enter a number between 1 and 40")
            except ValueError:
                print("Please enter a valid number")

        print("\nSearching for mentors...")
        mentors = await finder.find_potential_mentors(
            field=field,
            location=location if location else None,
            min_experience=min_experience,
        )

        if not mentors:
            print("\nNo mentors found. This could be due to:")
            print("1. API not properly configured (see instructions above)")
            print("2. No results matching your criteria")
            print("3. Network connectivity issues")
            print("\nPlease verify API setup or try different search criteria.")
            return

        print("\nTop Potential Mentors:")
        for i, mentor in enumerate(mentors, 1):
            print(f"\n{i}. {mentor['name']}")
            print(f"   Title: {mentor['title']}")
            print(f"   Expertise: {', '.join(mentor['expertise'])}")
            print(f"   Experience: {mentor.get('experience_years', 'N/A')} years")
            print(f"   LinkedIn: {mentor['profile_url']}")
            print(f"   Relevance Score: {mentor.get('relevance_score', 0):.2f}")

    except ValueError as ve:
        print(f"\nConfiguration Error: {str(ve)}")
        print("\nPlease follow these steps to set up your API:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable Custom Search API")
        print("4. Create credentials (API key)")
        print("5. Go to https://programmablesearchengine.google.com/")
        print("6. Create a new search engine")
        print("7. Get your Search Engine ID")
        print("\nThen set your environment variables:")
        print("export GOOGLE_API_KEY=your_api_key_here")
        print("export GOOGLE_CSE_ID=your_search_engine_id_here")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please verify your API credentials and internet connection.")


if __name__ == "__main__":
    asyncio.run(main())
