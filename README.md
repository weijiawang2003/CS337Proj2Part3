
# CS337 Project 2 – Part 3  
## Hybrid Recipe Assistant

---

## 1. What this project does

This project is a **command-line cooking assistant** that helps a user walk through an online recipe.

It is intentionally **hybrid**:

- **Python rules** handle anything that should be predictable and stable.
- A **Gemini LLM** is used only when flexible language understanding or cooking knowledge is actually helpful.

In other words: Python takes care of structure and state; the LLM fills in the “smart” explanations on top.

---

## 2. Main components

- `assistant.py`  
  Main entry point for the **hybrid assistant** (Part 3).  
  - Loads a recipe via `scrape.py`  
  - Tracks the current step  
  - Handles most commands with rules  
  - Calls the LLM only when needed

- `scrape.py`  
  Given a recipe URL, scrapes the page and returns a **structured JSON** recipe  
  (title, ingredients, steps, tools, methods, time, temperature).

- `parser.py`  
  Rule-based recipe parser used by `scrape.py`. Reused from earlier parts so the hybrid
  system has a clean, consistent representation to work with.

- `prompt.py`  
  Contains the `SYSTEM_PROMPT` used when talking to Gemini, including instructions about
  using the provided recipe and state.

- `interface.py`  
  The earlier **rule-only** assistant from Parts 1–2. Kept as a baseline to compare with
  the hybrid approach.

- `gemini_test.py`  
  Small script to verify that the Gemini API key and client are set up correctly.

---

## 3. Setup

1. Install dependencies (example):

   ```bash
   pip install google-genai python-dotenv requests beautifulsoup4 urllib3
````

2. Set your Gemini API key. For example, in a `.env` file in the project root:

   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

   or set the environment variable directly in your shell.

---

## 4. How to run

From the project directory, run:

```bash
python assistant.py
```

You’ll see something like:

```text
Hybrid LLM + rules-based recipe assistant (Part 3).
Paste a recipe URL:
```

Paste a recipe URL (e.g., an AllRecipes link).

The assistant scrapes and parses the recipe, shows how many steps there are, and starts at step 1.
You can then interact using natural commands:

### Navigation (rule-based)

* `next`, `next step`, `continue`
* `back`, `previous`, `go back`
* `step 3` or just `3` to jump to a specific step

### Ingredients and basics (rule-based)

* `ingredients`, `show ingredients`
* `how long do I bake it?`
* `what temp?`
* `how much sugar do I need?`

These are answered directly from the structured recipe JSON without calling the LLM.

### “How-to” help (hybrid with YouTube)

* `how do I preheat the oven?`
* `how do I do that?` (asked after a step)

For these, the assistant:

* Builds a short, context-aware search query from the current step
  (methods, ingredients, tools).
* Returns a YouTube search URL.
* Optionally adds an LLM explanation of what to do.

### Open-ended questions (LLM-backed)

For example:

* `why do we cut the rolls first?`
* `can I use milk instead of cream?`

Here the assistant passes the full recipe JSON, the current step, and the conversation
so far to Gemini, and returns a concise, context-aware answer.

If the Gemini free-tier quota is exceeded, the assistant switches to a **rules-only mode**:
navigation and structured questions continue to work, and the program does not crash.

---

## 5. Hybrid design in one paragraph

The assistant keeps step tracking and core logic entirely in Python (via a `RecipeState` dataclass) and treats the LLM as a controlled “add-on” rather than the main controller. Navigation, ingredients, time, temperature, and most quantity questions are handled with simple rules. The LLM is called only when the user asks for something more open-ended or interpretive, and it always receives the structured recipe and current step so its answers stay grounded. This illustrates a practical hybrid approach: let deterministic code handle state and safety, and use the LLM where natural language understanding and cooking knowledge genuinely help.

```
```

