"""
Hybrid LLM + rules-based cooking assistant (Part 3 - Enhanced).

This version integrates:
- LLM-based recipe parsing (can fall back to rules-based)
- LLM-based intent classification
- LLM-based state tracking (with Python fallback)
- LLM-based question answering
- LLM-refined YouTube search queries
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from google import genai

# Import both parsing methods
from scrape import scrape_recipe  # Rules-based (fallback)
try:
    from llm_parser import parse_recipe_from_url_llm  # LLM-based
    LLM_PARSER_AVAILABLE = True
except ImportError:
    LLM_PARSER_AVAILABLE = False

# Import LLM components
from llm_intent import classify_intent
from llm_state_tracker import infer_current_step
from prompts import build_qa_prompt, build_youtube_refinement_prompt

import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")
client = genai.Client(api_key=api_key)

LLM_DISABLED = False
USE_LLM_PARSER = True  # Set to False to use rules-based parser
USE_LLM_STATE_TRACKING = True  # Set to False to use Python-only tracking


def youtube_search(query: str) -> str:
    """Return a YouTube search URL for a given query."""
    base_url = "https://www.youtube.com/results?search_query="
    cleaned = " ".join(query.strip().split())
    return f"{base_url}{quote_plus(cleaned)}"


def refine_youtube_query_with_llm(
    user_question: str,
    current_step: Dict[str, Any],
    recipe: Dict[str, Any]
) -> str:
    """Use LLM to refine YouTube search query."""
    if LLM_DISABLED:
        # Fallback: extract basic query
        if "how do i " in user_question.lower():
            return user_question.lower().replace("how do i ", "").strip()
        if "how to " in user_question.lower():
            return user_question.lower().replace("how to ", "").strip()
        return user_question
    
    try:
        prompt = build_youtube_refinement_prompt(user_question, current_step, recipe)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        refined = response.text.strip().strip('"').strip("'")
        return refined if refined else user_question
    except Exception:
        return user_question


@dataclass
class RecipeState:
    recipe: Dict[str, Any]
    current_step_idx: int = 0

    @property
    def steps(self) -> List[Dict[str, Any]]:
        return self.recipe.get("steps", [])

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def current_step(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.current_step_idx < self.num_steps:
            return self.steps[self.current_step_idx]
        return None

    def to_state_dict(self) -> Dict[str, Any]:
        """Structured state given to the LLM as context."""
        step = self.current_step or {}
        return {
            "current_step_index": self.current_step_idx,
            "num_steps": self.num_steps,
            "current_step": {
                "step_number": step.get("step_number"),
                "description": step.get("description"),
                "ingredients": step.get("ingredients", []),
                "tools": step.get("tools", []),
                "methods": step.get("methods", []),
                "time": step.get("time", {}),
                "temperature": step.get("temperature", {}),
            },
        }

    def go_to_step(self, step_number: int) -> str:
        """Go to 1-based step_number and return a human-readable message."""
        if self.num_steps == 0:
            return "No steps found in this recipe."

        idx = step_number - 1
        if idx < 0 or idx >= self.num_steps:
            return f"Step {step_number} is out of range. This recipe has {self.num_steps} steps."
        self.current_step_idx = idx
        return self.format_current_step()

    def next_step(self) -> str:
        if self.num_steps == 0:
            return "No steps found in this recipe."
        if self.current_step_idx + 1 >= self.num_steps:
            return "You are already at the last step."
        self.current_step_idx += 1
        return self.format_current_step()

    def prev_step(self) -> str:
        if self.num_steps == 0:
            return "No steps found in this recipe."
        if self.current_step_idx == 0:
            return "You are already at the first step."
        self.current_step_idx -= 1
        return self.format_current_step()

    def format_current_step(self) -> str:
        step = self.current_step
        if not step:
            return "No current step."
        num = step.get("step_number", self.current_step_idx + 1)
        desc = (step.get("description") or "").strip()
        return f"Step {num}: {desc}"

    def format_ingredients(self) -> str:
        ingredients = self.recipe.get("ingredients", [])
        if not ingredients:
            return "No parsed ingredients found."
        lines = ["Here are the ingredients:"]
        for ing in ingredients:
            name = ing.get("name", "ingredient")
            qty = ing.get("quantity")
            unit = ing.get("measurement")
            descriptor = ing.get("descriptor")
            prep = ing.get("preparation")
            pieces = []
            if qty is not None:
                pieces.append(str(qty))
            if unit:
                pieces.append(unit)
            pieces.append(name)
            if descriptor:
                pieces.append(f"({descriptor})")
            if prep:
                pieces.append(f"- {prep}")
            lines.append("  - " + " ".join(pieces))
        return "\n".join(lines)


# Rule-based answer functions (kept as fallback/scaffolding)
def answer_time_question(state: RecipeState) -> str:
    step = state.current_step or {}
    time_info = step.get("time") or {}
    duration = time_info.get("duration")
    if duration:
        return f"In this step, the time is {duration}."
    
    for s in state.steps:
        t = (s.get("time") or {}).get("duration")
        if t:
            return f"The recipe uses a cooking time of {t} in one of the steps."

    return "This recipe doesn't specify a clear time for this step."


def answer_temp_question(state: RecipeState, norm: str) -> str:
    step = state.current_step or {}
    temp_info = step.get("temperature") or {}

    wants_internal = any(
        phrase in norm
        for phrase in ["internal", "inside", "center", "thickest part", "reach", "chicken", "meat"]
    )
    wants_oven = "oven" in norm or "bake" in norm or "baking" in norm or "preheat" in norm

    value: Optional[str] = None

    if wants_internal:
        for key in ["internal", "center", "meat"]:
            if key in temp_info:
                value = temp_info[key]
                break

    if value is None and wants_oven:
        for key in ["oven", "cooking", "bake"]:
            if key in temp_info:
                value = temp_info[key]
                break

    if value is None:
        if temp_info:
            value = next(iter(temp_info.values()))

    if value is None:
        for s in state.steps:
            t_info = s.get("temperature") or {}
            if t_info:
                value = next(iter(t_info.values()))
                break

    if value is None:
        return "This recipe doesn't specify a clear temperature for this step."

    return f"The temperature in this step is {value}."


def answer_quantity_question(state: RecipeState, user: str) -> str:
    norm = user.lower()
    ingredients = state.recipe.get("ingredients", [])
    mentioned: List[Dict[str, Any]] = []

    for ing in ingredients:
        name = (ing.get("name") or "").lower().strip()
        if not name:
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", norm):
            mentioned.append(ing)

    if not mentioned:
        m = re.search(r"how (?:many|much)\s+([a-zA-Z]+)", norm)
        if m:
            target = m.group(1)
            for ing in ingredients:
                name = (ing.get("name") or "").lower()
                if not name:
                    continue
                if target in name or name in target:
                    mentioned.append(ing)

    def format_ing(ing: Dict[str, Any]) -> str:
        name = ing.get("name", "ingredient")
        qty = ing.get("quantity")
        unit = ing.get("measurement")
        descriptor = ing.get("descriptor")
        prep = ing.get("preparation")
        parts = []
        if qty is not None:
            parts.append(str(qty))
        if unit:
            parts.append(unit)
        parts.append(name)
        if descriptor:
            parts.append(f"({descriptor})")
        if prep:
            parts.append(f"- {prep}")
        return " ".join(parts)

    if mentioned:
        if len(mentioned) == 1:
            return "You need " + format_ing(mentioned[0]) + "."
        else:
            lines = [f"- {format_ing(ing)}" for ing in mentioned]
            return "Here are the quantities:\n" + "\n".join(lines)

    step = state.current_step or {}
    step_ings = step.get("ingredients") or []
    if step_ings:
        lines = []
        for name in step_ings:
            name_low = name.lower()
            for ing in ingredients:
                if (ing.get("name") or "").lower() == name_low:
                    lines.append(f"- {format_ing(ing)}")
                    break
        if lines:
            return "In this step, you need:\n" + "\n".join(lines)

    return (
        "I'm not sure which ingredient you mean. Try asking something like "
        "'How much sugar do I need?'"
    )


def ask_llm(state: RecipeState, user_message: str, history: List[str] | None = None) -> str:
    """Ask LLM a question with full context."""
    global LLM_DISABLED

    if LLM_DISABLED:
        return (
            "I can't call the language model right now because the API quota "
            "or limit has been exhausted. You can still use commands like "
            "'next', 'back', and 'step 3' to navigate the recipe."
        )

    if history is None:
        history = []

    prompt = build_qa_prompt(
        recipe=state.recipe,
        current_state=state.to_state_dict(),
        conversation_history=history,
        user_question=user_message
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        return (response.text or "").strip()

    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower() or "429" in msg:
            LLM_DISABLED = True
            return (
                "I can't call the language model right now because the API quota "
                "has been exhausted."
            )

        return (
            "I ran into an error while calling the language model, so I couldn't "
            "answer that question."
        )


def update_state_from_dialogue(
    state: RecipeState,
    conversation_history: List[str]
) -> None:
    """Update state using LLM-based state tracking."""
    if not USE_LLM_STATE_TRACKING or LLM_DISABLED:
        return  # Keep Python-tracked state
    
    try:
        result = infer_current_step(
            recipe=state.recipe,
            conversation_history=conversation_history,
            last_known_step=state.current_step_idx
        )
        
        # Only update if confidence is high or medium
        if result["confidence"] in ["high", "medium"]:
            inferred_idx = result["current_step_index"]
            if 0 <= inferred_idx < state.num_steps:
                state.current_step_idx = inferred_idx
    except Exception:
        pass  # Keep current state on error


def main() -> None:
    print("Hybrid LLM + rules-based recipe assistant (Part 3 - Enhanced).")
    if USE_LLM_PARSER and LLM_PARSER_AVAILABLE:
        print("Using LLM parser: Enabled")
    else:
        print("Using LLM parser: Disabled (fallback to rules-based)")
    print(f"Using LLM state tracking: {USE_LLM_STATE_TRACKING}")
    
    url = input("Paste a recipe URL: ").strip()
    if not url:
        print("No URL provided, exiting.")
        return

    # Parse recipe (LLM or rules-based)
    try:
        if USE_LLM_PARSER and LLM_PARSER_AVAILABLE:
            print("Parsing recipe with LLM...")
            recipe_json = parse_recipe_from_url_llm(url)
            recipe = recipe_json if isinstance(recipe_json, dict) else json.loads(recipe_json)
        else:
            print("Parsing recipe with rules-based parser...")
            recipe_json = scrape_recipe(url)
            recipe = json.loads(recipe_json)
    except Exception as e:
        print(f"Failed to parse recipe: {e}")
        print("Falling back to rules-based parser...")
        try:
            recipe_json = scrape_recipe(url)
            recipe = json.loads(recipe_json)
        except Exception as e2:
            print(f"Failed to parse recipe: {e2}")
            return

    state = RecipeState(recipe=recipe, current_step_idx=0)
    history: List[str] = []

    title = recipe.get("title", "Unknown recipe")
    print(f'\nLoaded recipe: "{title}" with {state.num_steps} steps.')
    if state.num_steps:
        print(state.format_current_step())
    print(
        "\nType 'next', 'back', 'step N', 'N' (just a number), 'ingredients', "
        "or ask a question about the current step.\n"
        "Type 'quit' or 'exit' to end.\n"
    )

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        norm = user_message.lower().strip()

        # Exit
        if norm in {"quit", "exit"}:
            print("Assistant: Goodbye and happy cooking!")
            break

        # Classify intent using LLM
        intent_result = classify_intent(user_message, "\n".join(history[-4:]) if history else None)
        intent = intent_result.get("intent", "question")
        needs_llm = intent_result.get("needs_llm", True)
        needs_youtube = intent_result.get("needs_youtube", False)
        refined_query = intent_result.get("refined_query")

        # Override: If question starts with "how do I", "how to", "how should I", 
        # it's ALWAYS a how_to question, regardless of classification
        if any(phrase in norm for phrase in ["how do i", "how to", "how should i"]):
            intent = "how_to"
            needs_youtube = True
            needs_llm = True

        # Handle navigation (can be rule-based or LLM-tracked)
        if intent == "navigation":
            if "next" in norm or "continue" in norm:
                reply = state.next_step()
                print(f"Assistant: {reply}\n")
                history.append(f"User: {user_message}")
                history.append(f"Assistant: {reply}")
                continue

            if "back" in norm or "previous" in norm or "prev" in norm:
                reply = state.prev_step()
                print(f"Assistant: {reply}\n")
                history.append(f"User: {user_message}")
                history.append(f"Assistant: {reply}")
                continue

            m = re.search(r"\bstep\s+(\d+)\b", norm)
            if m:
                step_number = int(m.group(1))
                reply = state.go_to_step(step_number)
                print(f"Assistant: {reply}\n")
                history.append(f"User: {user_message}")
                history.append(f"Assistant: {reply}")
                continue

            if norm.isdigit():
                step_number = int(norm)
                reply = state.go_to_step(step_number)
                print(f"Assistant: {reply}\n")
                history.append(f"User: {user_message}")
                history.append(f"Assistant: {reply}")
                continue

        # Handle ingredients
        if intent == "ingredients":
            reply = state.format_ingredients()
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        # Handle how-to questions with YouTube FIRST (before temperature/time checks)
        youtube_link: Optional[str] = None
        if intent == "how_to" or needs_youtube:
            if refined_query:
                query = refined_query
            else:
                query = refine_youtube_query_with_llm(
                    user_message,
                    state.current_step or {},
                    state.recipe
                )
            youtube_link = youtube_search(query)
            # For how-to questions, always use LLM to explain, then add YouTube link
            answer = ask_llm(state, user_message, history)
            if youtube_link:
                combined = f"{answer}\n\nHere's a YouTube search that might help: {youtube_link}"
            else:
                combined = answer or "I'm not sure how to answer that."
            print(f"Assistant: {combined}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {combined}")
            continue

        # Handle time/temp/quantity with rules (fast) or LLM (if ambiguous)
        # Only check these if NOT a how-to question
        if intent == "time" and not needs_llm:
            reply = answer_time_question(state)
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        if intent == "temperature" and not needs_llm:
            # Check if it's actually a "how do I" question about temperature
            if "how do i" in norm or "how to" in norm or "how should i" in norm:
                # This is a how-to question, not a temperature lookup
                youtube_link = youtube_search(refine_youtube_query_with_llm(
                    user_message, state.current_step or {}, state.recipe
                ))
                answer = ask_llm(state, user_message, history)
                if youtube_link:
                    combined = f"{answer}\n\nHere's a YouTube search that might help: {youtube_link}"
                else:
                    combined = answer
                print(f"Assistant: {combined}\n")
                history.append(f"User: {user_message}")
                history.append(f"Assistant: {combined}")
                continue
            reply = answer_temp_question(state, norm)
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        if intent == "quantity" and not needs_llm:
            reply = answer_quantity_question(state, user_message)
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        # Handle out-of-scope
        if intent == "out_of_scope":
            reply = "I'm only able to answer questions about this recipe and cooking."
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        # Use LLM for everything else
        answer = ask_llm(state, user_message, history)

        # Combine answer with YouTube link if applicable
        if youtube_link:
            combined = f"{answer}\n\nHere's a YouTube search that might help: {youtube_link}"
        else:
            combined = answer or "I'm not sure how to answer that."

        print(f"Assistant: {combined}\n")

        history.append(f"User: {user_message}")
        history.append(f"Assistant: {combined}")

        # Update state from dialogue (LLM-based tracking)
        if USE_LLM_STATE_TRACKING:
            update_state_from_dialogue(state, history)


if __name__ == "__main__":
    main()

