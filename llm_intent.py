"""
LLM-based intent classifier for Part 3.
Replaces regex/keyword-based intent detection with an LLM classifier.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")
client = genai.Client(api_key=api_key)


INTENT_PROMPT = """You are an intent classifier for a cooking assistant. Classify the user's message into one of these categories:

INTENT_CATEGORIES:
1. "navigation" - Commands to navigate steps (next, back, step N, continue, previous, etc.)
2. "ingredients" - Questions about ingredients (show ingredients, ingredient list, etc.)
3. "time" - Questions about cooking time (how long, cooking time, baking time, etc.)
4. "temperature" - Questions about temperature (what temp, what temperature, how hot, etc.)
5. "quantity" - Questions about ingredient quantities (how much, how many, amount of, etc.)
6. "substitution" - Questions about substitutions (substitute for, instead of, replacement, etc.)
7. "how_to" - Questions about how to do something (how do I, how to, how should I, etc.)
8. "what_is" - Questions asking what something is (what is X, what is that, etc.)
9. "question" - General questions about the recipe, cooking techniques, or context
10. "acknowledgment" - Simple acknowledgments (yes, ok, got it, thanks, etc.)
11. "out_of_scope" - Questions unrelated to cooking or the recipe

Return a JSON object with:
{
  "intent": "one of the categories above",
  "confidence": "high" | "medium" | "low",
  "needs_llm": true | false,  // whether this needs LLM reasoning
  "needs_youtube": true | false,  // whether this should trigger YouTube search
  "refined_query": "refined search query for YouTube if applicable" | null
}

Rules:
- "navigation" intents can be handled by rules, set needs_llm to false
- "time", "temperature", "quantity" can often be handled by rules, but set needs_llm to true if ambiguous
- IMPORTANT: "how_to" takes priority over "temperature", "time", etc. If the question starts with "how do I", "how to", "how should I", it is ALWAYS "how_to", not "temperature" or "time"
- Examples: "how do I preheat an oven?" -> "how_to" (NOT "temperature"), "how long do I cook this?" -> "how_to" (NOT "time")
- "how_to" should set needs_youtube to true and provide a refined_query
- "question" always needs_llm true
- "out_of_scope" should be detected and needs_llm true to provide a polite rejection
- Be strict about out_of_scope - only mark as such if clearly unrelated to cooking/recipe
"""


def classify_intent(user_message: str, conversation_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify user intent using LLM.
    
    Args:
        user_message: The user's input message
        conversation_context: Optional conversation history for context
    
    Returns:
        Dictionary with intent classification results
    """
    context_part = ""
    if conversation_context:
        context_part = f"\n\nConversation context:\n{conversation_context}"
    
    prompt = f"""{INTENT_PROMPT}

User message: "{user_message}"{context_part}

Classify the intent and return ONLY valid JSON (no markdown, no code blocks):"""

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
        
        result = json.loads(response_text)
        
        # Validate structure
        if "intent" not in result:
            result["intent"] = "question"
        if "confidence" not in result:
            result["confidence"] = "medium"
        if "needs_llm" not in result:
            result["needs_llm"] = True
        if "needs_youtube" not in result:
            result["needs_youtube"] = False
        if "refined_query" not in result:
            result["refined_query"] = None
        
        return result
        
    except json.JSONDecodeError:
        # Fallback to basic classification
        return {
            "intent": "question",
            "confidence": "low",
            "needs_llm": True,
            "needs_youtube": False,
            "refined_query": None,
        }
    except Exception as e:
        # Fallback on error
        return {
            "intent": "question",
            "confidence": "low",
            "needs_llm": True,
            "needs_youtube": False,
            "refined_query": None,
        }


if __name__ == "__main__":
    test_messages = [
        "next step",
        "how long do I bake it?",
        "what temperature?",
        "how much sugar do I need?",
        "how do I preheat the oven?",
        "what is a whisk?",
        "can I use milk instead of cream?",
        "what's the weather today?",
    ]
    
    for msg in test_messages:
        result = classify_intent(msg)
        print(f"Message: {msg}")
        print(f"Intent: {result['intent']}, Needs LLM: {result['needs_llm']}, Needs YouTube: {result['needs_youtube']}")
        print()

