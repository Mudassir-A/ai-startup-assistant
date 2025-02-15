import requests
import pandas as pd
from typing import List, Dict, Any
from transformers import pipeline
from serpapi import GoogleSearch
import asyncio
import aiohttp
from datetime import datetime
import logging
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class MentorFinder:
    def __init__(self, serpapi_key: str):
        """Initialize the MentorFinder with necessary API keys and models."""
        self.api_key = serpapi_key
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
        """
        Find potential mentors based on field and criteria.
        """
        try:
            # Create search queries
            queries = [
                f"{field} entrepreneur founder",
                f"{field} startup CEO",
                f"{field} business mentor",
                f"{field} industry expert",
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
        """
        Search for potential mentors using Google Search API.
        """
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": 10,
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            mentors = []
            for result in results.get("organic_results", []):
                mentor_info = await self._extract_mentor_info(session, result)
                if mentor_info:
                    mentors.append(mentor_info)

            return mentors

        except Exception as e:
            self.logger.error(f"Error in mentor search: {str(e)}")
            return []

    async def _extract_mentor_info(
        self, session: aiohttp.ClientSession, result: Dict
    ) -> Dict[str, Any]:
        """
        Extract mentor information from search result.
        """
        try:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")

            # Skip if not likely to be about a person
            if not self._is_likely_person(title, snippet):
                return None

            # Extract name and basic info
            name = self._extract_name(title)
            if not name:
                return None

            # Analyze the content
            profile_info = {
                "name": name,
                "title": title,
                "summary": snippet,
                "profile_url": link,
                "expertise": self._extract_expertise(snippet),
                "experience_years": self._extract_experience(snippet),
                "contact_info": self._extract_contact_info(snippet),
                "sentiment_score": self._analyze_sentiment(snippet),
                "source": "search_result",
                "last_updated": datetime.now().isoformat(),
            }

            return profile_info

        except Exception as e:
            self.logger.error(f"Error extracting mentor info: {str(e)}")
            return None

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

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """
        Extract contact information from text.
        """
        contact_info = {}

        # Extract email
        email_indicators = ["email", "contact", "@"]
        for indicator in email_indicators:
            if indicator in text.lower():
                # Simple email pattern matching
                words = text.split()
                for word in words:
                    if "@" in word and "." in word:
                        contact_info["email"] = word.strip(".,;()")
                        break

        # Extract LinkedIn profile
        if "linkedin.com" in text.lower():
            start_idx = text.lower().find("linkedin.com")
            end_idx = text.find(" ", start_idx)
            if end_idx == -1:
                end_idx = len(text)
            contact_info["linkedin"] = text[start_idx:end_idx].strip(".,;()")

        return contact_info

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
    # Initialize with your SerpAPI key
    finder = MentorFinder("your_serpapi_key_here")

    # Example usage
    field = input("Enter your field of interest: ")
    location = input("Enter preferred location (or press Enter to skip): ")
    min_experience = int(input("Enter minimum years of experience required: "))

    print("\nSearching for mentors...")
    mentors = await finder.find_potential_mentors(
        field=field,
        location=location if location else None,
        min_experience=min_experience,
    )

    print("\nTop Potential Mentors:")
    for i, mentor in enumerate(mentors, 1):
        print(f"\n{i}. {mentor['name']}")
        print(f"   Expertise: {', '.join(mentor['expertise'])}")
        print(f"   Experience: {mentor.get('experience_years', 'N/A')} years")
        print(f"   Profile: {mentor.get('profile_url', 'N/A')}")
        if mentor.get("contact_info"):
            print(f"   Contact: {mentor['contact_info']}")
        print(f"   Relevance Score: {mentor.get('relevance_score', 0):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
