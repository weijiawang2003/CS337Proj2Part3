"""
LLM-based recipe parser for Part 3.
Replaces the rules-based parser with an LLM that can extract structured recipe data
from HTML or raw text.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")
client = genai.Client(api_key=api_key)


PARSER_PROMPT = """You are a recipe parser. Your task is to extract structured recipe information from HTML or text content.

Given recipe content, extract and return a JSON object with the following structure:
{
  "title": "Recipe title",
  "ingredients": [
    {
      "name": "ingredient name",
      "quantity": 1.5,  // number or null
      "measurement": "cup",  // unit like "cup", "tablespoon", "oz", etc. or null
      "descriptor": "fresh",  // optional descriptor like "fresh", "chopped", etc. or null
      "preparation": "chopped"  // optional preparation instruction or null
    }
  ],
  "steps": [
    {
      "step_number": 1,
      "description": "Step description text",
      "ingredients": ["ingredient name"],  // list of ingredient names used in this step
      "tools": ["tool name"],  // list of tools/equipment needed
      "methods": ["method name"],  // list of cooking methods (bake, boil, chop, etc.)
      "time": {
        "duration": "30 minutes"  // time duration or null
      },
      "temperature": {
        "oven": "350°F",  // or "cooking": "medium heat", "internal": "165°F", etc.
      }
    }
  ],
  "tools": ["list of all unique tools used in recipe"],
  "methods": ["list of all unique cooking methods used"]
}

Important:
- Extract ingredients with quantities, units, descriptors, and preparation when present
- Break down steps into atomic actions (one main action per step)
- Extract tools, methods, time, and temperature from each step
- Be precise with quantities (handle fractions like 1/2, 1/4, etc.)
- Return valid JSON only, no markdown formatting
"""


def extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML, prioritizing recipe content."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Try to find recipe-specific content
    recipe_sections = soup.find_all(["article", "div"], class_=lambda x: x and any(
        keyword in x.lower() for keyword in ["recipe", "ingredient", "instruction", "step"]
    ))
    
    if recipe_sections:
        text_parts = []
        for section in recipe_sections:
            text_parts.append(section.get_text(separator="\n", strip=True))
        return "\n\n".join(text_parts)
    
    # Fallback to body text
    return soup.get_text(separator="\n", strip=True)


def parse_recipe_with_llm(content: str, is_html: bool = True) -> Dict[str, Any]:
    """
    Parse a recipe using LLM.
    
    Args:
        content: HTML content or plain text
        is_html: Whether content is HTML (True) or plain text (False)
    
    Returns:
        Dictionary with structured recipe data
    """
    if is_html:
        # Extract text from HTML first
        text_content = extract_text_from_html(content)
    else:
        text_content = content
    
    # Limit content size to avoid token limits
    if len(text_content) > 10000:
        text_content = text_content[:10000] + "..."
    
    prompt = f"""{PARSER_PROMPT}

Recipe content to parse:
{text_content}

Extract the recipe information and return ONLY valid JSON (no markdown, no code blocks, just the JSON object):"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        recipe_data = json.loads(response_text)
        
        # Validate and normalize structure
        if "steps" not in recipe_data:
            recipe_data["steps"] = []
        if "ingredients" not in recipe_data:
            recipe_data["ingredients"] = []
        if "tools" not in recipe_data:
            recipe_data["tools"] = []
        if "methods" not in recipe_data:
            recipe_data["methods"] = []
        
        # Ensure steps have required fields
        for step in recipe_data["steps"]:
            if "ingredients" not in step:
                step["ingredients"] = []
            if "tools" not in step:
                step["tools"] = []
            if "methods" not in step:
                step["methods"] = []
            if "time" not in step:
                step["time"] = {}
            if "temperature" not in step:
                step["temperature"] = {}
        
        # Add gather step if not present
        if recipe_data["steps"]:
            first_step_desc = recipe_data["steps"][0].get("description", "").lower()
            if "gather" not in first_step_desc:
                gather_step = {
                    "step_number": 1,
                    "description": "Gather all ingredients.",
                    "ingredients": [ing.get("name", "").lower() for ing in recipe_data["ingredients"]],
                    "tools": [],
                    "methods": ["gather"],
                    "time": {},
                    "temperature": {},
                }
                # Renumber existing steps
                for step in recipe_data["steps"]:
                    step["step_number"] = step.get("step_number", 1) + 1
                recipe_data["steps"] = [gather_step] + recipe_data["steps"]
        
        return recipe_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text}")
    except Exception as e:
        raise ValueError(f"LLM parsing failed: {e}")


def parse_recipe_from_url_llm(url: str) -> Dict[str, Any]:
    """
    Parse a recipe from a URL using LLM.
    
    Args:
        url: Recipe URL
    
    Returns:
        Dictionary with structured recipe data
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html = response.text
        return parse_recipe_with_llm(html, is_html=True)
    except Exception as e:
        raise ValueError(f"Failed to fetch or parse recipe from URL: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python llm_parser.py <recipe_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    recipe = parse_recipe_from_url_llm(url)
    print(json.dumps(recipe, indent=2, ensure_ascii=False))

