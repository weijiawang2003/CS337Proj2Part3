"""
LLM-based state tracker for Part 3.
Infers the current recipe step from dialogue history.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")
client = genai.Client(api_key=api_key)


STATE_TRACKER_PROMPT = """You are a state tracker for a cooking assistant. Your task is to infer the current step the user is on based on the conversation history and recipe structure.

Given:
- The recipe with all steps
- The conversation history
- The last known step index (if any)

Determine what step the user is currently on.

Return a JSON object with:
{
  "current_step_index": <0-based index>,  // -1 if cannot determine
  "confidence": "high" | "medium" | "low",
  "reasoning": "brief explanation of why this step"
}

Rules:
- Step indices are 0-based (first step is 0)
- Look for explicit step mentions ("step 3", "go to step 2", etc.)
- Track navigation commands ("next", "back", "continue", "previous")
- If user asks about a specific step's content, they might be on that step
- If conversation just started, user is likely on step 0
- If user says "next" after step 3, they're now on step 4
- If user says "back" after step 5, they're now on step 4
- Be conservative - if uncertain, return the last known step or step 0
"""


def infer_current_step(
    recipe: Dict[str, Any],
    conversation_history: List[str],
    last_known_step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Infer the current step from conversation history.
    
    Args:
        recipe: The recipe dictionary with steps
        conversation_history: List of conversation turns (alternating user/assistant)
        last_known_step: Last known step index (if tracking externally)
    
    Returns:
        Dictionary with inferred step index and confidence
    """
    steps_info = []
    for step in recipe.get("steps", []):
        steps_info.append({
            "step_number": step.get("step_number", 0),
            "description": step.get("description", "")[:100],  # Truncate for context
        })
    
    history_text = "\n".join(conversation_history[-10:])  # Last 10 turns
    
    prompt = f"""{STATE_TRACKER_PROMPT}

Recipe steps:
{json.dumps(steps_info, indent=2)}

Conversation history:
{history_text}

Last known step index: {last_known_step if last_known_step is not None else "unknown"}

Infer the current step and return ONLY valid JSON (no markdown, no code blocks):"""

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
        
        # Validate and clamp step index
        step_idx = result.get("current_step_index", 0)
        num_steps = len(recipe.get("steps", []))
        if step_idx < 0:
            step_idx = 0
        if step_idx >= num_steps and num_steps > 0:
            step_idx = num_steps - 1
        
        result["current_step_index"] = step_idx
        
        if "confidence" not in result:
            result["confidence"] = "medium"
        if "reasoning" not in result:
            result["reasoning"] = "Inferred from conversation"
        
        return result
        
    except (json.JSONDecodeError, Exception):
        # Fallback: use last known step or default to 0
        fallback_idx = last_known_step if last_known_step is not None else 0
        num_steps = len(recipe.get("steps", []))
        if fallback_idx >= num_steps and num_steps > 0:
            fallback_idx = num_steps - 1
        if fallback_idx < 0:
            fallback_idx = 0
        
        return {
            "current_step_index": fallback_idx,
            "confidence": "low",
            "reasoning": "Fallback due to parsing error",
        }


if __name__ == "__main__":
    # Test with sample recipe and conversation
    test_recipe = {
        "steps": [
            {"step_number": 1, "description": "Gather all ingredients."},
            {"step_number": 2, "description": "Preheat oven to 350°F."},
            {"step_number": 3, "description": "Mix ingredients in a bowl."},
        ]
    }
    
    test_history = [
        "User: start recipe",
        "Assistant: Step 1: Gather all ingredients.",
        "User: next",
        "Assistant: Step 2: Preheat oven to 350°F.",
    ]
    
    result = infer_current_step(test_recipe, test_history)
    print(f"Inferred step: {result['current_step_index'] + 1}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reasoning: {result['reasoning']}")

