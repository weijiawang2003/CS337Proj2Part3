from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from google import genai

from scrape import scrape_recipe
from prompt import SYSTEM_PROMPT

import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)



load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")
client = genai.Client(api_key=api_key)

LLM_DISABLED = False


def youtube_search(query: str) -> str:
    """
    Return a YouTube search URL for a given short query.

    We normalize whitespace and use '+' for spaces.
    """
    base_url = "https://www.youtube.com/results?search_query="
    cleaned = " ".join(query.strip().split())
    return f"{base_url}{quote_plus(cleaned)}"



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
        """
        Structured state given to the LLM as context.
        """
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



def is_time_question(norm: str) -> bool:
    if "how long" in norm or "for how long" in norm:
        return True
    if "cooking time" in norm or "baking time" in norm:
        return True
    if "time" in norm and ("cook" in norm or "bake" in norm or "simmer" in norm):
        return True
    return False


def is_temp_question(norm: str) -> bool:
    if norm in {
        "temp", "temperature",
        "what temp", "what temperature",
        "what degree", "what degrees",
        "what heat",
    }:
        return True

    if "temperature" in norm or "temp" in norm or "degrees" in norm:
        return True
    if "what heat" in norm or "how hot" in norm:
        return True
    if "bake at" in norm or "baked at" in norm:
        return True
    if "preheat" in norm and ("to" in norm or "at" in norm):
        return True

    return False


def is_quantity_question(norm: str) -> bool:
    if "how much" in norm or "how many" in norm:
        return True
    if "amount of" in norm or "quantity of" in norm:
        return True
    return False



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
    global LLM_DISABLED

    if LLM_DISABLED:
        return (
            "I can't call the language model right now because the API quota "
            "or limit has been exhausted. You can still use commands like "
            "'next', 'back', and 'step 3' to navigate the recipe."
        )

    if history is None:
        history = []

    conversation_text = ""
    for turn in history:
        conversation_text += turn + "\n\n"

    payload = {
        "recipe": state.recipe,
        "state": state.to_state_dict(),
        "conversation_so_far": conversation_text,
        "user_message": user_message,
    }

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "You are now in HYBRID MODE:\n"
        "- Python code is tracking the current step of the recipe.\n"
        "- You receive the full recipe JSON and a small state object.\n"
        "- Use this information to answer the user's question accurately.\n"
        "- Do NOT invent steps that are impossible given the recipe.\n"
        "- If the question is clearly off-topic (not about cooking or this recipe), "
        "say: 'I am only able to answer questions about this recipe and cooking.'\n\n"
        "Here is the recipe and state context as JSON:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "Now answer the user's last question. Be concise (2–4 sentences).\n"
        "User:\n"
        f"{user_message}\n"
        "Assistant:"
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





PRONOUN_PHRASES = [
    "how do i do that",
    "how do i do this",
    "how do i do it",
]


def build_contextual_howto_query(state: RecipeState) -> str:
    """
    Build a short how-to query from the current step.
    """
    step = state.current_step or {}
    methods = step.get("methods", [])
    ingredients = step.get("ingredients", [])
    tools = step.get("tools", [])

    verb: Optional[str] = None
    if methods:
        verb = methods[0]

    obj: Optional[str] = None
    if ingredients:
        obj = ingredients[0]

    if verb and obj:
        base = f"how to {verb} {obj}"
    elif verb:
        base = f"how to {verb}"
    elif obj:
        base = f"how to use {obj}"
    else:
        base = "basic cooking technique"

    if tools:
        tool = tools[0]
        if not re.search(r"\bbowl\b", tool):
            base = f"{base} in {tool}"
    return base


def extract_direct_howto_query(norm: str) -> str:
    if norm.startswith("how do i "):
        core = norm[len("how do i "):]
    elif norm.startswith("how to "):
        core = norm[len("how to "):]
    else:
        core = norm

    core = re.sub(r"\b(in|for|with|on|in this recipe|for this recipe|here)\b.*$", "", core)
    core = core.strip()

    if core in {"do that", "do this", "do it"} or not core:
        return ""

    return f"how to {core}"


ACKS = {
    "yes", "yeah", "yep", "ok", "okay", "sure",
    "sounds good", "alright", "got it", "thanks", "thank you",
}


def main() -> None:
    print("Hybrid LLM + rules-based recipe assistant (Part 3).")
    url = input("Paste a recipe URL: ").strip()
    if not url:
        print("No URL provided, exiting.")
        return


    recipe_json = scrape_recipe(url)
    try:
        recipe = json.loads(recipe_json)
    except json.JSONDecodeError:
        print("Failed to parse recipe JSON from scrape.py.")
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

        # --- exit ---
        if norm in {"quit", "exit"}:
            print("Assistant: Goodbye and happy cooking!")
            break


        if "what step" in norm and "on" in norm:
            reply = state.format_current_step()
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue


        if norm in ACKS:
            reply = (
                "Got it. When you're ready for the next step, type 'next'.\n"
                + state.format_current_step()
            )
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue



        if norm in {"next", "next step", "continue"} or norm.startswith("next "):
            reply = state.next_step()
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        if norm in {"back", "previous", "prev", "go back"} or norm.startswith("back "):
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


        if norm in {
            "ingredients",
            "ingredient list",
            "show ingredients",
            "show me the ingredients",
        }:
            reply = state.format_ingredients()
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        if is_time_question(norm):
            reply = answer_time_question(state)
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        if is_temp_question(norm):
            reply = answer_temp_question(state, norm)
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        if is_quantity_question(norm):
            reply = answer_quantity_question(state, user_message)
            print(f"Assistant: {reply}\n")
            history.append(f"User: {user_message}")
            history.append(f"Assistant: {reply}")
            continue

        youtube_link: Optional[str] = None
        is_howto = norm.startswith("how do i ") or norm.startswith("how to ")
        has_pronoun = any(p in norm for p in PRONOUN_PHRASES)

        if is_howto or has_pronoun:
            query = extract_direct_howto_query(norm)
            if not query or has_pronoun:
                query = build_contextual_howto_query(state)
            youtube_link = youtube_search(query)


        answer = ask_llm(state, user_message, history)

        if youtube_link:
            combined = f"{answer}\n\nHere's a YouTube search that might help: {youtube_link}"
        else:
            combined = answer or "I'm not sure how to answer that."

        print(f"Assistant: {combined}\n")

        history.append(f"User: {user_message}")
        history.append(f"Assistant: {combined}")


if __name__ == "__main__":
    main()
