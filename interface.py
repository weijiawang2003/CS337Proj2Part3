from __future__ import annotations
from typing import Optional, Any
import string
import re
from urllib.parse import quote

from parser import parse_recipe_from_url, Recipe, Step



try:
    import speech_recognition as sr
except ImportError:
    sr = None

EXIT_COMMANDS = ["quit", "exit","end","goodbye","bye"]
QUIT_MESSAGE = "Bot: Goodbye!"



class RecipeBot:
    def __init__(self, use_speech: bool = False):
        self.recipe: Optional[Recipe] = None
        self.current_step_idx: int = 0
        self.use_speech = use_speech and (sr is not None)
        self.recognizer: Optional[Any] = None

        if use_speech and sr is None:
            print("Bot: speech_recognition is not installed; falling back to text input.")
            self.use_speech = False
        elif use_speech:
            self.recognizer = sr.Recognizer()

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
        return text




    def run(self) -> None:
        print("Bot: Hi! I can walk you through a recipe from AllRecipes.com.")
        if self.use_speech:
            print("Bot: Speech input is ENABLED. Say 'quit' to exit.")
        else:
            print("Bot: Speech input is DISABLED. Type 'quit' to exit.")

        print("Bot: Please paste or say a recipe URL to get started.")

        while True:
            user = self.get_user_input()
            if user is None:
                continue

            if not user:
                continue

            norm = self.normalize(user)
            if norm in EXIT_COMMANDS:
                print(QUIT_MESSAGE)
                break

            response = self.handle_input(user)
            print(f"Bot: {response}")

    def get_user_input(self) -> Optional[str]:
        if not self.use_speech:
            try:
                return input("User: ").strip()
            except KeyboardInterrupt:
                print(f"\n{QUIT_MESSAGE}")
                return "quit"

        if self.recognizer is None:
            self.recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                print("User (speak): ", end="", flush=True)
                audio = self.recognizer.listen(source)

            try:
                text = self.recognizer.recognize_google(audio)
                print(text)
                return text.strip()
            except sr.UnknownValueError:
                print("\nBot: Sorry, I didn't catch that. Please repeat.")
                return None
            except sr.RequestError as e:
                print(f"\nBot: STT service error ({e}). Falling back to keyboard input.")
                self.use_speech = False
                self.recognizer = None
                return input("User: ").strip()
        except KeyboardInterrupt:
            print(f"\n{QUIT_MESSAGE}")
            return "quit"

    def handle_input(self, user: str) -> str:
        raw = user.strip()
        norm = self.normalize(raw)

        if raw.startswith("http"):
            return self.load_recipe(raw)

        if self.recipe is None:
            if "allrecipes" in norm:
                url = raw
                if not url.startswith("http"):
                    url = "https://" + url
                return self.load_recipe(url)
            return "Please paste or say an AllRecipes.com URL first."

        if norm in [
            "1", "ingredients", "ingredient list", "show me the ingredients list",
            "show ingredients", "go over ingredients", "go over ingredients list"
        ]:
            return self.answer_ingredients_question(norm)

        if norm in [
            "2", "steps", "go over steps", "start steps",
            "show steps", "go over recipe steps"
        ]:
            self.current_step_idx = 0
            return self.show_current_step()

        if self.is_next_command(norm):
            return self.next_step()

        if self.is_back_command(norm):
            return self.prev_step()

        if self.is_repeat_command(norm):
            return self.show_current_step()

        
        if "first step" in norm or "go to step one" in norm or "go to the first step" in norm:
            self.current_step_idx = 0
            return self.show_current_step()
        
        step_num = self._extract_step_number(norm)
        if step_num is not None:
            return self.go_to_step(step_num)


        if "last step" in norm or "final step" in norm or "go to the end" in norm:
            return self.go_to_last_step()

        if self.is_steps_overview_question(norm):
            return self.answer_steps_overview()


        if "ingredient" in norm and not self.is_quantity_question(norm):
            return self.answer_ingredients_question(norm)

        if "tool" in norm or "equipment" in norm:
            return self.answer_tools_question(norm)


        if self.is_time_question(norm):
            return self.answer_time_question()

        if self.is_temp_question(norm):
            return self.answer_temp_question(norm)

        if self.is_quantity_question(norm):
            return self.answer_quantity_question(raw)

        if self.is_substitution_question(norm):
            return self.answer_substitution_question(norm)

        if (
            "how do i do that" in norm
            or "how do i do this" in norm
            or "how do i do it" in norm
        ):
            return self.answer_vague_how_to()
        
        
        if norm.startswith("what is "):
            return self.answer_what_is(norm)
        
        if norm.startswith("how do i "):
            query = norm[len("how do i "):].strip()
            return f"https://www.youtube.com/results?search_query=how+to+{quote(query)}"
        
        if norm.startswith("how to "):
            query = norm[len("how to "):].strip()
            return f"https://www.youtube.com/results?search_query=how+to+{quote(query)}"
        
        return (
            "I didn't quite catch that.\n"
            "You can try commands like:\n"
            "- '1' or 'show me the ingredients list'\n"
            "- '2' or 'go over steps'\n"
            "- 'next step', 'go to the next step', 'continue'\n"
            "- 'go back one step', 'previous step'\n"
            "- 'repeat that', 'say that again'\n"
            "- 'How long do I bake it for?'\n"
            "- 'What temperature should the oven be?'\n"
            "- 'How many eggs do I need?', 'How much salt do I need?'\n"
            "- 'What is a whisk?' or 'What is it?'\n"
            "- 'How do I knead the dough?' or 'How do I do that?'"
        )

    @staticmethod
    def is_next_command(norm: str) -> bool:
        patterns = [
            "next step", "go to the next step", "go to next step",
            "next", "continue", "whats next", "what is next"
        ]
        return any(p in norm for p in patterns)

    @staticmethod
    def is_back_command(norm: str) -> bool:
        patterns = [
            "go back one step", "go back a step", "go back",
            "previous step", "previous", "back"
        ]
        return any(p in norm for p in patterns)

    @staticmethod
    def is_repeat_command(norm: str) -> bool:
        patterns = [
            "repeat please", "repeat that", "say that again", "again", "repeat"
        ]
        return any(p in norm for p in patterns)

    @staticmethod
    def is_time_question(norm: str) -> bool:
        if "how long" in norm or "for how long" in norm:
            return True
        if "cooking time" in norm or "baking time" in norm:
            return True
        if "time" in norm and ("cook" in norm or "bake" in norm or "simmer" in norm):
            return True
        return False


    @staticmethod
    def is_temp_question(norm: str) -> bool:
        if norm in {
            "temp", "temperature",
            "what temp", "what temperature",
            "what degree", "what degrees",
            "what heat"
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



    @staticmethod
    def is_quantity_question(norm: str) -> bool:
        if "how much" in norm or "how many" in norm:
            return True
        if "amount of" in norm or "quantity of" in norm:
            return True
        return False


    @staticmethod
    def is_steps_overview_question(norm: str) -> bool:
        if "how many steps" in norm or "number of steps" in norm:
            return True
        if "show all steps" in norm or "list all steps" in norm or "show me all steps" in norm:
            return True
        return False


    @staticmethod
    def is_substitution_question(norm: str) -> bool:
        if "substitute for" in norm or "what can i substitute" in norm:
            return True
        if "what can i use instead" in norm or "use instead" in norm:
            return True
        if "instead of" in norm or "replacement for" in norm or "alternative to" in norm:
            return True
        return False


    @staticmethod
    def format_ingredient(ing) -> str:
        q = f"{ing.quantity:g} " if ing.quantity is not None else ""
        unit = f"{ing.unit} " if ing.unit else ""
        desc = f"{ing.descriptor} " if ing.descriptor else ""
        prep = f", {ing.preparation}" if ing.preparation else ""
        return f"{q}{unit}{desc}{ing.name}{prep}"



    def load_recipe(self, url: str) -> str:
        if not url or not url.strip():
            return "Please provide a valid URL."

        # Basic URL validation
        if not (url.startswith("http://") or url.startswith("https://")):
            return "Please provide a valid URL starting with http:// or https://"

        try:
            self.recipe = parse_recipe_from_url(url)
            self.current_step_idx = 0
            return (
                f"Alright. So let's start working with \"{self.recipe.title}\".\n"
                "What do you want to do?\n"
                "[1] Go over ingredients list\n"
                "[2] Go over recipe steps."
            )
        except ValueError as e:
            return f"Could not parse the recipe from that URL: {e}"
        except Exception as e:
            return f"Something went wrong loading that recipe: {e}"

    def show_ingredients(self) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."
        lines = [f'Here are the ingredients for "{self.recipe.title}":']
        for ing in self.recipe.ingredients:
            lines.append(f"- {self.format_ingredient(ing)}")
        return "\n".join(lines)

    def get_current_step(self) -> Step:
        if self.recipe is None:
            raise ValueError("No recipe loaded. Please load a recipe first.")
        if not self.recipe.steps:
            raise ValueError("Recipe has no steps.")
        if self.current_step_idx >= len(self.recipe.steps):
            raise ValueError(f"Step index {self.current_step_idx} out of range.")
        return self.recipe.steps[self.current_step_idx]

    def show_current_step(self) -> str:
        try:
            step = self.get_current_step()
            ordinal = self.ordinal(step.step_number)
            return f"The {ordinal} step is: {step.description}"
        except ValueError as e:
            return str(e)

    def next_step(self) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."
        if self.current_step_idx + 1 >= len(self.recipe.steps):
            return "You are at the last step."
        self.current_step_idx += 1
        return self.show_current_step()

    def prev_step(self) -> str:
        if self.current_step_idx == 0:
            return "You are at the first step."
        self.current_step_idx -= 1
        return self.show_current_step()

    @staticmethod
    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    

    def go_to_step(self, n: int) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."
        if n < 1 or n > len(self.recipe.steps):
            return f"This recipe has {len(self.recipe.steps)} steps; step {n} does not exist."
        self.current_step_idx = n - 1
        return self.show_current_step()

    def go_to_last_step(self) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."
        self.current_step_idx = len(self.recipe.steps) - 1
        return self.show_current_step()


    @staticmethod
    def _extract_step_number(norm: str) -> Optional[int]:
        m = re.search(r"step\s*(\d+)", norm)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        m = re.search(r"(\d+)(st|nd|rd|th)?\s+step", norm)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None




    def answer_time_question(self) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."
        try:
            step = self.get_current_step()
        except ValueError as e:
            return str(e)

        if step.time.get("duration"):
            return f"In this step, the time is {step.time['duration']}."

        verbs = set(step.methods)
        for s in self.recipe.steps:
            if s.time.get("duration"):
                if verbs & set(s.methods):
                    return (
                        f"For {', '.join(verbs)} earlier, "
                        f"the recipe says: {s.time['duration']}."
                    )


        for s in self.recipe.steps:
            if s.time.get("duration"):
                return f"Earlier, the recipe says: {s.time['duration']}."
        return "The recipe does not specify a clear time here."




    def answer_temp_question(self, norm: str) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."

        try:
            step = self.get_current_step()
        except ValueError as e:
            return str(e)

        cooking_temp = step.temperature.get("cooking")
        internal_temp = step.temperature.get("internal")
        context_temp = step.context.get("cooking_temperature")

        wants_internal = any(
            phrase in norm
            for phrase in ["internal", "inside", "center", "thickest part", "reach", "chicken", "meat"]
        )
        wants_oven = "oven" in norm or "bake" in norm or "baking" in norm or "roast" in norm

        if wants_internal and (internal_temp or internal_temp is not None):
            if internal_temp:
                return f"In this step, the internal temperature should be {internal_temp}."
            for s in self.recipe.steps:
                if s.temperature.get("internal"):
                    return f"The internal temperature should be {s.temperature['internal']}."


        if wants_oven:
            if cooking_temp:
                return f"In this step, the cooking temperature is {cooking_temp}."
            if context_temp:
                return f"The cooking temperature should be {context_temp}."

        if cooking_temp:
            return f"In this step, the cooking temperature is {cooking_temp}."
        if context_temp:
            return f"The cooking temperature should be {context_temp}."
        if internal_temp:
            return f"In this step, the internal temperature should be {internal_temp}."


        for s in self.recipe.steps:
            temp = s.temperature.get("cooking") or s.context.get("cooking_temperature")
            if temp:
                return f"The recipe uses a cooking temperature of {temp}."
        for s in self.recipe.steps:
            temp = s.temperature.get("internal")
            if temp:
                return f"The recipe specifies an internal temperature of {temp}."

        return "The recipe does not specify a temperature here."





    def _extract_quantity_target_phrase(self, norm: str) -> Optional[str]:
        m = re.search(
            r"(how much|how many)\s+([a-z ]+?)(\s+(do i|do we|do you|should i|should we|should you)\b|\?|$)",
            norm,
        )
        if m:
            phrase = m.group(2).strip()
            return phrase if phrase else None
        return None

    def answer_quantity_question(self, user: str) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."
        norm = self.normalize(user)

        mentioned = []

        for ing in self.recipe.ingredients:
            name = ing.name.lower()
            if not name:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", norm):
                mentioned.append(ing)

        if not mentioned:
            target_phrase = self._extract_quantity_target_phrase(norm)
            if target_phrase:
                target_tokens = [t for t in target_phrase.split() if t]
                for ing in self.recipe.ingredients:
                    name_low = ing.name.lower()
                    if not name_low:
                        continue
                    for tok in target_tokens:
                        tok = tok.strip()
                        if not tok:
                            continue
                        if tok in name_low or name_low in tok:
                            mentioned.append(ing)
                            break

        if mentioned:
            lines = [f"- {self.format_ingredient(ing)}" for ing in mentioned]
            if len(lines) == 1:
                return f"You need {lines[0][2:]}."
            else:
                return "Here are the quantities:\n" + "\n".join(lines)

        
        step = self.get_current_step()
        if step.ingredients:
            lines = []
            for name in step.ingredients:
                for ing in self.recipe.ingredients:
                    if ing.name.lower() == name:
                        lines.append(f"- {self.format_ingredient(ing)}")
                        break
            if lines:
                if len(lines) == 1:
                    return f"For that, you need {lines[0][2:]}."
                else:
                    return "For this step, the relevant quantities are:\n" + "\n".join(lines)

        return "I'm not sure which ingredient you mean."

    def answer_vague_how_to(self) -> str:
        step = self.get_current_step()

        verb: Optional[str] = None
        if step.action:
            verb = step.action
        elif step.methods:
            verb = step.methods[0]
        obj: Optional[str] = None
        
        if step.ingredients:
            obj = step.ingredients[0]

        if verb and obj:
            query = f"how to {verb} {obj}"
            return f"https://www.youtube.com/results?search_query={quote(query)}"

        if verb:
            query = f"how to {verb}"
            return f"https://www.youtube.com/results?search_query={quote(query)}"

        if obj:
            query = f"how to use {obj}"
            return f"https://www.youtube.com/results?search_query={quote(query)}"

        return "I'm not sure what 'that' refers to in this step."

    def answer_what_is(self, norm: str) -> str:
        body = norm[len("what is "):].strip()

        if self.recipe is None:
            query = body.replace(" ", "+")
            return f"https://www.google.com/search?q=what+is+{query}"
        if body and body not in {"it", "that", "this"}:
            query = body.replace(" ", "+")
            return f"https://www.google.com/search?q=what+is+{query}"
        step = self.get_current_step()
        
        if step.methods:
            method = step.methods[0]
            query = f"what is {method} in cooking".replace(" ", "+")
            return f"https://www.google.com/search?q={query}"
        if step.tools:
            tool = step.tools[0]
            query = f"what is a {tool}".replace(" ", "+")
            return f"https://www.google.com/search?q={query}"
        if step.ingredients:
            ing_name = step.ingredients[-1]
            query = f"what is {ing_name}".replace(" ", "+")
            return f"https://www.google.com/search?q={query}"

        
        return "I'm not sure what 'that' refers to here. Could you be more specific?"




    def answer_ingredients_question(self, norm: str) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."

        step = self.get_current_step()

        if "all ingredient" in norm or "all the ingredients" in norm or "entire recipe" in norm or "whole recipe" in norm:
            return self.show_ingredients()

        if step.step_number == 1 and "this step" not in norm and "current step" not in norm:
            return self.show_ingredients()


        if "this step" in norm or "current step" in norm or "for this step" in norm:
            names = step.ingredients
            if not names:
                return "This step does not list any specific ingredients."
            lines = []
            for name in names:
                for ing in self.recipe.ingredients:
                    if ing.name.lower() == name:
                        lines.append(f"- {self.format_ingredient(ing)}")
                        break
            if not lines:
                return "I couldn't find ingredient details for this step."
            return "For this step, you need:\n" + "\n".join(lines)

        if step.step_number > 1:
            names = step.ingredients
            if not names:
                return "This step does not list any specific ingredients."
            lines = []
            for name in names:
                for ing in self.recipe.ingredients:
                    if ing.name.lower() == name:
                        lines.append(f"- {self.format_ingredient(ing)}")
                        break
            if not lines:
                return "I couldn't find ingredient details for this step."
            return "For this step, you need:\n" + "\n".join(lines)

        return self.show_ingredients()

    def answer_tools_question(self, norm: str) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."

        step = self.get_current_step()

        if "this step" in norm or "current step" in norm or "for this step" in norm:
            if not step.tools:
                return "This step doesn't need any special tools—just standard kitchen items."
            return "For this step, you need: " + ", ".join(step.tools) + "."
        else:
            if not self.recipe.tools:
                return "No specific tools were detected for this recipe."
            return "For this recipe, you need: " + ", ".join(self.recipe.tools) + "."

    def answer_steps_overview(self) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."
        total = len(self.recipe.steps)
        lines = [f"This recipe has {total} steps.", "Here are all the steps:"]
        for step in self.recipe.steps:
            lines.append(f"{step.step_number}. {step.description}")
        return "\n".join(lines)

    def answer_substitution_question(self, norm: str) -> str:
        if self.recipe is None:
            return "No recipe loaded. Please load a recipe first."
        
        patterns = [
            r"substitute for ([a-z ]+)",
            r"instead of ([a-z ]+)",
            r"replacement for ([a-z ]+)",
            r"alternative to ([a-z ]+)",
            r"what can i use instead of ([a-z ]+)",
        ]

        target_phrase: Optional[str] = None
        for pat in patterns:
            m = re.search(pat, norm)
            if m:
                target_phrase = m.group(1).strip()
                break

        ingredient_name: Optional[str] = None

        if target_phrase:
            for ing in self.recipe.ingredients:
                name_low = ing.name.lower()
                if target_phrase in name_low or name_low in target_phrase:
                    ingredient_name = ing.name
                    break
            
            if ingredient_name is None:
                ingredient_name = target_phrase


        if ingredient_name is None:
            step = self.get_current_step()
            if step.ingredients:
                ingredient_name = step.ingredients[-1]

        if ingredient_name is None:
            return "I'm not sure which ingredient you want to substitute."

        query = f"substitute for {ingredient_name}".replace(" ", "+")
        return f"https://www.google.com/search?q={query}"


if __name__ == "__main__":
    use_speech_choice = input("Enable speech input? (y/n): ").strip().lower()
    use_speech = use_speech_choice.startswith("y")
    bot = RecipeBot(use_speech=use_speech)
    bot.run()
