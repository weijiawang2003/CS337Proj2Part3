"""
Enhanced prompts for different LLM functionalities in the hybrid system.
"""

import json

# System prompt for general question answering
SYSTEM_PROMPT_QA = """You are a helpful cooking assistant helping a user follow a recipe.

You have access to:
- The full recipe with ingredients, steps, tools, and methods
- The current step the user is on
- The conversation history

Your role:
- Answer questions accurately based on the recipe context
- Provide helpful cooking advice when appropriate
- Stay focused on the recipe and cooking-related topics
- Be concise (2-4 sentences typically)
- If asked about something not in the recipe, use general cooking knowledge but note it's not recipe-specific

Important:
- Do NOT invent steps or ingredients that aren't in the recipe
- If information isn't available, say so clearly
- If the question is off-topic (not about cooking or this recipe), politely redirect:
  "I'm only able to answer questions about this recipe and cooking."
- Keep responses relevant and helpful
"""

# Prompt for question answering with context
def build_qa_prompt(
    recipe: dict,
    current_state: dict,
    conversation_history: list,
    user_question: str
) -> str:
    """Build a prompt for question answering."""
    history_text = "\n".join(conversation_history[-6:]) if conversation_history else "No previous conversation."
    
    return f"""{SYSTEM_PROMPT_QA}

Recipe:
{json.dumps(recipe, indent=2, ensure_ascii=False)}

Current Step State:
{json.dumps(current_state, indent=2, ensure_ascii=False)}

Conversation History:
{history_text}

User Question: {user_question}

Answer the question based on the recipe context. Be concise and accurate."""


# Prompt for YouTube search query refinement
YOUTUBE_REFINEMENT_PROMPT = """You are helping refine a search query for YouTube cooking videos.

Given a user's question about how to do something in a recipe, create a concise, effective YouTube search query.

Rules:
- Keep it short (3-6 words typically)
- Focus on the main action/technique
- Include relevant context (tool, ingredient) if it makes the search more specific
- Remove recipe-specific details that won't help in a general search
- Use natural language that people would search for

Examples:
- "how do I preheat the oven for this lasagna?" -> "how to preheat oven"
- "how do I chop onions for this step?" -> "how to chop onions"
- "how do I knead dough?" -> "how to knead dough"
- "how do I do that?" (context: current step mentions "sear the meat") -> "how to sear meat"

Return ONLY the refined search query, nothing else."""


def build_youtube_refinement_prompt(
    user_question: str,
    current_step: dict,
    recipe_context: dict
) -> str:
    """Build a prompt to refine YouTube search query."""
    step_desc = current_step.get("description", "")[:200] if current_step else ""
    
    return f"""{YOUTUBE_REFINEMENT_PROMPT}

User question: "{user_question}"

Current step context:
{step_desc}

Recipe methods/tools: {', '.join(recipe_context.get('methods', [])[:5])}

Create a refined YouTube search query:"""

