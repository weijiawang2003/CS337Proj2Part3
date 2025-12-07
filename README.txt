CS337 Project 2 - Part 3: Hybrid Recipe Assistant
==================================================

GitHub Repository: [Please add your GitHub link here]

SETUP
-----

1. Install dependencies:
   pip install google-genai python-dotenv requests beautifulsoup4 urllib3 nltk

2. Download NLTK data (if using rules-based parser):
   python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

3. Set your Gemini API key in a .env file:
   GEMINI_API_KEY=your_api_key_here

   Or set it as an environment variable:
   export GEMINI_API_KEY=your_api_key_here

4. Run the assistant:
   python assistant_hybrid.py

USAGE
-----

1. Start the program: python assistant_hybrid.py

2. Paste a recipe URL (e.g., from AllRecipes.com)

3. The system will parse the recipe and show the first step

4. Interact with commands:
   - Navigation: "next", "back", "step 3", "3"
   - Ingredients: "ingredients", "show ingredients"
   - Questions: "how long do I bake it?", "what temperature?", "how much sugar?"
   - How-to: "how do I preheat the oven?" (triggers YouTube search)
   - General: "can I use milk instead of cream?", "why do we cut the rolls first?"

MODEL AND SETTINGS
------------------

Model: gemini-2.5-flash-lite
Temperature: Default (not explicitly set, uses model default)
Max tokens: Not explicitly set (uses model default)

The system uses gemini-2.5-flash-lite for all LLM operations:
- Recipe parsing
- Intent classification
- State tracking
- Question answering
- YouTube query refinement

ARCHITECTURE
------------

The hybrid system combines rule-based components with LLM modules:

1. RECIPE PARSING (llm_parser.py)
   - Primary: LLM-based parser that extracts structured recipe data from HTML/text
   - Fallback: Rules-based parser (parser.py) if LLM parsing fails
   - Extracts: title, ingredients (with quantities/units), steps (with tools, methods, time, temperature)
   - Output: Structured JSON matching the recipe representation

2. INTENT CLASSIFICATION (llm_intent.py)
   - Replaces regex/keyword matching with LLM-based classification
   - Categories: navigation, ingredients, time, temperature, quantity, substitution, how_to, what_is, question, acknowledgment, out_of_scope
   - Returns: intent category, confidence, needs_llm flag, needs_youtube flag, refined_query
   - Enables smart routing to appropriate handlers

3. STATE TRACKING (llm_state_tracker.py)
   - Infers current recipe step from dialogue history
   - Uses LLM to understand implicit step changes
   - Falls back to Python-based tracking if LLM unavailable
   - Maintains RecipeState dataclass for structured state representation

4. QUESTION ANSWERING (prompts.py + assistant_hybrid.py)
   - Uses LLM with full recipe context, current step, and conversation history
   - Handles open-ended questions, substitutions, explanations
   - Grounded in recipe data to reduce hallucinations
   - Concise responses (2-4 sentences typically)

5. YOUTUBE SEARCH INTEGRATION
   - Detects "how-to" questions via intent classification
   - Uses LLM to refine/generate search queries from user questions
   - Context-aware: uses current step information to improve queries
   - Returns YouTube search URLs alongside LLM answers

6. SCAFFOLDING COMPONENTS (assistant_hybrid.py)
   - Rule-based handlers for fast, predictable operations:
     * Navigation (next, back, step N)
     * Time/temperature/quantity extraction from structured data
     * Ingredient listing
   - Python state management (RecipeState class)
   - Separation of concerns: control logic separate from LLM reasoning
   - Fallback mechanisms when LLM unavailable

COMPONENT IMPLEMENTATION DETAILS
----------------------------------

Recipe Parser (llm_parser.py):
- Extracts text from HTML using BeautifulSoup
- Sends recipe content to LLM with structured extraction prompt
- Parses JSON response and validates structure
- Adds "gather ingredients" step if not present
- Handles errors gracefully with fallback

Intent Classifier (llm_intent.py):
- Single-shot classification per user message
- Uses conversation context for better classification
- Returns structured JSON with intent and metadata
- Fallback to "question" intent on errors

State Tracker (llm_state_tracker.py):
- Analyzes conversation history to infer current step
- Considers explicit mentions, navigation commands, step references
- Returns step index with confidence level
- Only updates state if confidence is high/medium

Question Answerer:
- Builds comprehensive prompt with recipe, state, and history
- Uses SYSTEM_PROMPT_QA for consistent behavior
- Handles out-of-scope questions with polite redirection
- Grounded in recipe data to prevent hallucinations

YouTube Query Refiner:
- Takes user question and current step context
- Generates concise, effective search queries
- Removes recipe-specific details that won't help in general search
- Returns refined query for YouTube search function

Hybrid Design Principles:
- Python handles: state management, navigation, structured data extraction
- LLM handles: parsing, intent classification, state inference, open-ended questions
- Fallbacks: Rules-based parser, Python state tracking, error handling
- Efficiency: Use rules for fast operations, LLM for complex reasoning
- Separation: Control logic in Python, reasoning in LLM with clear boundaries

INTENT CATEGORIES AND MAPPING
------------------------------

1. navigation -> Python rule-based handlers (next, back, step N)
2. ingredients -> Python rule-based handler (format_ingredients)
3. time -> Python rule-based handler (if unambiguous) or LLM (if ambiguous)
4. temperature -> Python rule-based handler (if unambiguous) or LLM (if ambiguous)
5. quantity -> Python rule-based handler (if unambiguous) or LLM (if ambiguous)
6. substitution -> LLM (requires cooking knowledge)
7. how_to -> LLM + YouTube search (requires explanation and video link)
8. what_is -> LLM (requires definition/explanation)
9. question -> LLM (general questions about recipe/cooking)
10. acknowledgment -> Python rule-based handler (simple response)
11. out_of_scope -> LLM (polite rejection message)

EFFICIENCY CONSIDERATIONS
-------------------------

- Uses gemini-2.5-flash-lite (faster, cheaper model) for all operations
- Rules-based handlers for common, predictable operations (navigation, simple Q&A)
- LLM only called when needed (intent classification determines routing)
- Conversation history limited to last 6-10 turns to reduce context size
- Recipe content truncated if too long (>10000 chars for parsing)

ERROR HANDLING
--------------

- LLM parsing failures fall back to rules-based parser
- LLM API quota exhaustion disables LLM but keeps rules-based functionality
- Intent classification errors fall back to "question" intent
- State tracking errors keep current Python-tracked state
- All errors handled gracefully without crashing

TESTING
-------

Test with various recipe URLs from AllRecipes.com:
- Simple recipes (few ingredients, few steps)
- Complex recipes (many ingredients, many steps)
- Recipes with unusual formatting

Test question types:
- Navigation commands
- Time/temperature/quantity questions
- Substitution questions
- How-to questions (should trigger YouTube)
- Out-of-scope questions (should be rejected politely)
- General cooking questions

FILES
-----

- assistant_hybrid.py: Main entry point with hybrid system
- llm_parser.py: LLM-based recipe parser
- llm_intent.py: LLM-based intent classifier
- llm_state_tracker.py: LLM-based state tracker
- prompts.py: Enhanced prompts for different LLM functionalities
- scrape.py: Rules-based recipe parser (fallback)
- parser.py: Rules-based parsing utilities
- prompt.py: Original system prompt (kept for compatibility)

NOTES
-----

- The system is designed to be robust: if LLM fails, rules-based components continue working
- State tracking can be disabled (USE_LLM_STATE_TRACKING = False) to use Python-only tracking
- LLM parser can be disabled (USE_LLM_PARSER = False) to use rules-based parser only
- All LLM components use the same model (gemini-2.5-flash-lite) for consistency
- Responses are kept concise to reduce token usage and improve user experience

