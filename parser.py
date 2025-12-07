from __future__ import annotations
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import json
import re

import requests
from bs4 import BeautifulSoup
import argparse
import nltk
from nltk.tokenize import sent_tokenize

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag as _nltk_pos_tag, word_tokenize as _nltk_word_tokenize
except Exception:
    wn = None
    WordNetLemmatizer = None
    _nltk_pos_tag = None
    _nltk_word_tokenize = None


@dataclass
class Ingredient:
    raw: str
    name: str
    quantity: Optional[float]
    unit: Optional[str]
    descriptor: Optional[str]
    preparation: Optional[str]


@dataclass
class Step:
    step_number: int
    description: str   
    ingredients: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    time: Dict[str, str] = field(default_factory=dict)
    temperature: Dict[str, str] = field(default_factory=dict)
    action: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    modifiers: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, str] = field(default_factory=dict)


@dataclass
class Recipe:
    title: str
    url: str
    ingredients: List[Ingredient]
    tools: List[str]
    methods: List[str]
    steps: List[Step]




UNITS = [
    "#",
    "#s",
    "bag",
    "bags",
    "bottle",
    "bottles",
    "bunch",
    "bunches",
    "c",
    "can",
    "cans",
    "clove",
    "cloves",
    "cs",
    "cube",
    "cubes",
    "cup",
    "cups",
    "dash",
    "dashes",
    "dessertspoon",
    "dessertspoons",
    "envelope",
    "envelopes",
    "fl oz",
    "fl ozs",
    "fluid ounce",
    "fluid ounces",
    "fluid oz",
    "fluid ozs",
    "gal",
    "gallon",
    "gallons",
    "gals",
    "gram",
    "grams",
    "head",
    "heads",
    "inch",
    "inches",
    "jar",
    "jars",
    "kilogram",
    "kilograms",
    "lb",
    "lbs",
    "liter",
    "liters",
    "milliliter",
    "milliliters",
    "ml",
    "mls",
    "ounce",
    "ounces",
    "oz",
    "ozs",
    "package",
    "packages",
    "packet",
    "packets",
    "piece",
    "pieces",
    "pinch",
    "pinches",
    "pint",
    "pints",
    "pound",
    "pounds",
    "pt",
    "pts",
    "qt",
    "qts",
    "quart",
    "quarts",
    "sheet",
    "sheets",
    "slice",
    "slices",
    "strip",
    "strips",
    "tablespoon",
    "tablespoons",
    "Tbsp",
    "Tbsps",
    "teaspoon",
    "teaspoons",
    "tsp",
    "tsps",
]

UNITS = {u.lower() for u in UNITS}

DESCRIPTORS = [
    "fresh",
    "freshly",
    "frozen",
    "chilled",
    "cold",
    "cool",
    "warm",
    "lukewarm",
    "hot",
    "room temperature",
    "refrigerated",
    "thawed",
    "dried",
    "dry",
    "dry roasted",
    "tender",
    "tough",
    "soft",
    "firm",
    "chewy",
    "crunchy",
    "crispy",
    "crumbly",
    "crusty",
    "fluffy",
    "dense",
    "smooth",
    "creamy",
    "syrupy",
    "bubbly",
    "sweet",
    "sweetened",
    "unsweetened",
    "bitter",
    "salty",
    "savory",
    "acidic",
    "tangy",
    "sour",
    "spicy",
    "mild",
    "fiery",
    "earthy",
    "smoky",
    "rich",
    "buttery",
    "lean",
    "meaty",
    "fatty",
    "nonfat",
    "low sodium",
    "reduced sodium",
    "unsalted",
    "extra virgin",
    "rare",
    "medium rare",
    "medium-rare",
    "medium",
    "well done",
    "gamey",
    "juicy",
    "whole",
    "halved",
    "quartered",
    "cubed",
    "diced",
    "minced",
    "chopped",
    "finely chopped",
    "coarsely chopped",
    "sliced",
    "thinly sliced",
    "thick-cut",
    "julienned",
    "matchstick cut",
    "shredded",
    "crumbled",
    "crushed",
    "ground",
    "mashed",
    "peeled",
    "pitted",
    "seeded",
    "cored",
    "torn",
    "roughly chopped",
    "baked",
    "boiled",
    "blanched",
    "braised",
    "broiled",
    "browned",
    "charred",
    "caramelized",
    "fried",
    "pan-fried",
    "deep-fried",
    "grilled",
    "roasted",
    "toasted",
    "steamed",
    "simmered",
    "stewed",
    "sauteed",
    "stir-fried",
    "canned",
    "candied",
    "smoked",
    "cured",
    "pickled",
    "fermented",
    "freeze dried",
    "dried out",
    "boneless",
    "bony",
    "skinless",
    "meaty",
    "moist",
    "juicy",
    "superfine",
    "organic",
    "natural",
    "fragrant",
    "aromatic",
    # Additional descriptors for better parsing
    "large",
    "small",
    "medium",
    "all-purpose",
    "all purpose",
    "seasoned",
]

DESCRIPTORS = {u.lower() for u in DESCRIPTORS}
DESCRIPTOR_WORDS = {w for d in DESCRIPTORS for w in d.split()}

PREPARATION_MAIN_VERBS = {
    "beaten",
    "zested",
    "chopped",
    "minced",
    "sliced",
    "diced",
    "peeled",
    "crushed",
    "grated",
    "ground",
    "shredded",
    "crumbled",
    "halved",
    "quartered",
    "trimmed",
    "seeded",
    "rinsed",
    "drained",
    "softened",
    "melted",
    "divided",
    "separated",
    "whisked",
    "beating",
    "cut",
}

PREPARATION_LEADING_ADVERBS = {
    "finely",
    "coarsely",
    "roughly",
    "lightly",
}

def looks_like_preparation_phrase(phrase: str) -> bool:
    words = phrase.strip().lower().split()
    if not words:
        return False
    first = words[0]
    if first in PREPARATION_MAIN_VERBS:
        return True
    if (
        len(words) >= 2
        and first in PREPARATION_LEADING_ADVERBS
        and words[1] in PREPARATION_MAIN_VERBS
    ):
        return True

    return False


TOOLS = [
    "knife",
    "chef knife",
    "chef's knife",
    "chefs knife",
    "paring knife",
    "bread knife",
    "utility knife",
    "carving knife",
    "steak knife",
    "santoku knife",
    "cutting board",
    "bench scraper",
    "dough scraper",
    "peeler",
    "vegetable peeler",
    "grater",
    "cheese grater",
    "nutmeg grater",
    "microplane",
    "zester",
    "nutcracker",
    "spoon",
    "wooden spoon",
    "slotted spoon",
    "spatula",
    "fish spatula",
    "burger spatula",
    "turner",
    "tongs",
    "silicone tong",
    "whisk",
    "balloon whisk",
    "flat whisk",
    "french whisk",
    "mixing whisk",
    "bowl",
    "mixing bowl",
    "cup",
    "measuring cup",
    "measuring jar",
    "measuring jug",
    "measuring spoon",
    "food storage container",
    "frying pan",
    "skillet",
    "grill pan",
    "griddle",
    "saucepan",
    "pot",
    "stockpot",
    "clay pot",
    "beanpot",
    "mated colander pot",
    "mortar",
    "molcajete",
    "baking sheet",
    "baking dish",
    "cake pan",
    "loaf pan",
    "muffin tin",
    "pie dish",
    "pie server",
    "pie cutter",
    "pizza cutter",
    "pizza shovel",
    "pizza slicer",
    "rolling pin",
    "pastry bag",
    "pastry brush",
    "pastry blender",
    "pastry wheel",
    "cookie cutter",
    "cookie mould",
    "cookie press",
    "biscuit cutter",
    "biscuit mould",
    "biscuit press",
    "colander",
    "sieve",
    "drum sieve",
    "strainer",
    "spider",
    "spider strainer",
    "spoon skimmer",
    "spoon sieve",
    "blender",
    "food mill",
    "food processor",
    "coffee grinder",
    "burr grinder",
    "burr mill",
    "milk frother",
    "garlic press",
    "citrus reamer",
    "lemon reamer",
    "lemon squeezer",
    "cherry pitter",
    "apple corer",
    "apple cutter",
    "mandoline",
    "ice cream scoop",
    "melon baller",
    "egg slicer",
    "egg separator",
    "thermometer",
    "meat thermometer",
    "candy thermometer",
    "kitchen scale",
    "weighing scales",
    "timer",
    "oven",
    "stove",
    "oven mitt",
    "oven glove",
    "pot holder",
    "potholder",
    "trussing needle",
    "kitchen twine",
    "cooking twine",
    "butcher's twine",
    "meat mallet",
]


TOOLS = {u.lower() for u in TOOLS}

PRIMARY_METHODS = [
    "bake",
    "boil",
    "broil",
    "fry",
    "deep-fry",
    "pan-fry",
    "stir-fry",
    "braise",
    "roast",
    "grill",
    "steam",
    "stew",
    "simmer",
    "saute",
    "searing",
    "pressure cook",
    "blend",
    "mix",
    "poach",
]

# FIX #5: Added missing cooking methods
OTHER_METHODS = [
    "beat",
    "brown",
    "brush",
    "chill",
    "combine",
    "cool",
    "cover",
    "cream",
    "crumble",
    "cut",
    "dip",
    "drain",
    "flip",
    "flour",
    "fold",
    "garnish",
    "grease",
    "heat",
    "layer",
    "line",
    "mash",
    "measure",
    "melt",
    "mix",
    "pound",      # Already present
    "pour",
    "preheat",
    "refrigerate",
    "rinse",
    "season",
    "serve",
    "shake",
    "sift",
    "simmer",
    "slice",
    "soak",
    "spoon",
    "spread",
    "sprinkle",
    "stir",
    "strain",
    "stuff",
    "toast",
    "toss",
    "turn",
    "whisk",
    "chop",
    "mince",
    "dice",
    "julienne",
    "shred",
    "grate",
    "knead",
    "marinate",
    # Added missing methods
    "dredge",
    "coat",
    "press",
    "arrange",
    "drizzle",
    "place",
    "remove",
    "repeat",
]










def _init_lemmatizer():
    if WordNetLemmatizer is None:
        return None
    try:
        return WordNetLemmatizer()
    except Exception:
        return None


_LEMMATIZER = _init_lemmatizer()

def _lemmatize(token: str, pos: str) -> str:

    t = token.lower()
    if _LEMMATIZER is not None and pos in {"n", "v"}:
        try:
            return _LEMMATIZER.lemmatize(t, pos)
        except Exception:
            pass


    if pos == "v":
        for suf in ["ing", "ed", "es", "s"]:
            if t.endswith(suf) and len(t) > len(suf) + 1:
                return t[: -len(suf)]
    
    if pos == "n" and t.endswith("s") and len(t) > 3:
        return t[:-1]
    return t


def _build_cooking_verbs_from_wordnet() -> List[str]:
    if wn is None:
        return []
    seeds = ["cook", "bake", "boil", "fry", "roast", "grill", "steam", "saute", "sauté"]
    verbs = set()
    
    for seed in seeds:
        try:
            for syn in wn.synsets(seed, pos="v"):
                for lemma in syn.lemmas():
                    verbs.add(lemma.name().replace("_", " ").lower())
        except Exception:
            break
    return sorted(verbs)


COOKING_VERBS = set(PRIMARY_METHODS + OTHER_METHODS)
COOKING_VERBS.update(_build_cooking_verbs_from_wordnet())


TOOL_LEMMA_TO_CANONICAL: Dict[str, str] = {}
for _tool in TOOLS:
    lemma = _lemmatize(_tool, "n")
    TOOL_LEMMA_TO_CANONICAL.setdefault(lemma, _tool)




def fetch_html(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text


def parse_allrecipes_basic(html: str) -> Dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    json_ld = soup.find("script", type="application/ld+json")
    if json_ld is None or not json_ld.string:
        raise ValueError("Could not find recipe JSON-LD on page.")

    data = json.loads(json_ld.string)
    if isinstance(data, list):
        recipe_obj = None
        for item in data:
            t = item.get("@type")
            if t == "Recipe" or (isinstance(t, list) and "Recipe" in t):
                recipe_obj = item
                break

        if recipe_obj is None:
            raise ValueError("JSON-LD does not contain a Recipe object.")
        data = recipe_obj

    title = data.get("name", "Unknown recipe")
    raw_ingredients = data.get("recipeIngredient", [])
    instructions = data.get("recipeInstructions", [])

    steps_raw: List[str] = []
    for inst in instructions:
        if isinstance(inst, dict):
            text = inst.get("text", "")
            if text:
                steps_raw.append(text.strip())
        elif isinstance(inst, str):
            if inst.strip():
                steps_raw.append(inst.strip())

    return {
        "title": title,
        "ingredients_raw": raw_ingredients,
        "steps_raw": steps_raw,
    }



def parse_quantity(token: str):
    token = token.strip()
    if len(token) == 1:
        try:
            return unicodedata.numeric(token)
        except (TypeError, ValueError):
            pass

    FRACTION_UNICODE = {
        "¼", "½", "¾",
        "⅐", "⅑", "⅒",
        "⅓", "⅔",
        "⅕", "⅖", "⅗", "⅘",
        "⅙", "⅚",
        "⅛", "⅜", "⅝", "⅞",
    }

    if len(token) > 1 and token[-1] in FRACTION_UNICODE:
        int_part = token[:-1]
        frac_char = token[-1]
        try:
            return float(int_part) + unicodedata.numeric(frac_char)
        except Exception:
            pass

    try:
        return float(token)
    except ValueError:
        pass

    if "/" in token:
        try:
            num, denom = token.split("/")
            return float(num) / float(denom)
        except Exception:
            pass

    return None




# FIX #4: Prioritize DESCRIPTORS check before POS tag for better ingredient parsing
def parse_ingredient_line(line: str) -> Ingredient:
    raw = line.strip()
    tokens = raw.split()
    quantity: Optional[float] = None
    unit: Optional[str] = None
    descriptor_tokens: List[str] = []
    name_tokens: List[str] = []
    preparation: Optional[str] = None


    if tokens:
        q = parse_quantity(tokens[0])
        if q is not None:
            quantity = q
            tokens = tokens[1:]



    if tokens:
        rest_lower = " ".join(tokens).lower()
        multiword_units = [u for u in UNITS if " " in u]
        multiword_units.sort(key=len, reverse=True)
        matched = False

        for u in multiword_units:
            if rest_lower.startswith(u + " ") or rest_lower == u:
                unit = u
                unit_len = len(u.split())
                tokens = tokens[unit_len:]
                matched = True
                break

        if not matched and tokens and tokens[0].lower() in UNITS:
            unit = tokens[0].lower()
            tokens = tokens[1:]


    before_comma, *after_comma = " ".join(tokens).split(",", 1)
    if after_comma:
        prep_str = after_comma[0].strip()
        if prep_str and looks_like_preparation_phrase(prep_str):
            preparation = prep_str


    descriptor_tokens = []
    name_tokens = []

    tagged = _pos_tag(before_comma)
    desc_pos = {"JJ", "RB", "VBN"}

    seen_noun = False
    for tok, tag in tagged:
        if not re.match(r"[A-Za-z]+$", tok):
            continue

        tok_lower = tok.lower()

        # FIX #4: Check DESCRIPTORS first before POS tag
        # This fixes "ground paprika" -> name: "paprika", descriptor: "ground"
        if tok_lower in DESCRIPTORS and not seen_noun:
            descriptor_tokens.append(tok_lower)
            continue

        if not seen_noun:
            if tag.startswith("NN"):
                seen_noun = True
                name_tokens.append(tok)
            # Only add as descriptor if BOTH: has descriptor POS AND is in DESCRIPTORS
            # This prevents "olive" from being a descriptor in "olive oil"
            elif tag in desc_pos and tok_lower in DESCRIPTORS:
                descriptor_tokens.append(tok_lower)
            else:
                # Not a noun and not a known descriptor - add to name
                # This keeps "olive" as part of "olive oil"
                name_tokens.append(tok)
                seen_noun = True  # Treat as start of name
        else:
            if tag.startswith("NN"):
                name_tokens.append(tok)
            else:
                if tag in desc_pos and tok_lower in DESCRIPTORS:
                    descriptor_tokens.append(tok_lower)


    if not name_tokens:
        for tok in before_comma.split():
            if tok.lower() in DESCRIPTORS:
                descriptor_tokens.append(tok.lower())
            else:
                name_tokens.append(tok)

    descriptor = " ".join(descriptor_tokens) if descriptor_tokens else None
    name = " ".join(name_tokens).strip()

    return Ingredient(
        raw=raw,
        name=name,
        quantity=quantity,
        unit=unit,
        descriptor=descriptor,
        preparation=preparation,
    )


def parse_ingredients(raw_ingredients: List[str]) -> List[Ingredient]:
    return [parse_ingredient_line(line) for line in raw_ingredients]




def split_into_atomic_steps(step_text: str) -> List[str]:
    parts = sent_tokenize(step_text)
    return [p.strip() for p in parts if p.strip()]


def find_items_in_text(text: str, vocab: List[str]) -> List[str]:
    if not vocab:
        return []
    text_lower = text.lower()
    pattern = r"\b(" + "|".join(sorted((re.escape(w.lower()) for w in vocab), key=len, reverse=True)) + r")\b"

    matches = re.findall(pattern, text_lower)
    found: List[str] = []
    seen = set()
    for m in matches:
        if m not in seen:
            seen.add(m)
            found.append(m)

    return found


def extract_time(text: str) -> Dict[str, str]:
    text_stripped = text.strip()
    range_match = re.search(
        r"\b(?:about|around|approximately|approx\.?)?\s*"
        r"(\d+)\s*(?:-|to)\s*(\d+)\s*"
        r"(minutes?|minute|min|mins?|hours?|hour|hrs?|seconds?|second|secs?)\b",
        text_stripped,
        flags=re.I,
    )
    
    if range_match:
        return {"duration": range_match.group(0).strip()}

    single_match = re.search(
        r"\b(?:about|around|approximately|approx\.?)?\s*"
        r"(\d+)\s*"
        r"(minutes?|minute|min|mins?|hours?|hour|hrs?|seconds?|second|secs?)\b",
        text_stripped,
        flags=re.I,
    )
    if single_match:
        return {"duration": single_match.group(0).strip()}

    return {}


# FIX #3: Temperature extraction distinguishes cooking temp vs internal temp
def extract_temperature(text: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    text_stripped = text.strip()
    text_lower = text_stripped.lower()

    # Check for internal temp indicators FIRST
    internal_indicators = [
        "thermometer", "internal temp", "internal temperature",
        "center", "thickest part", "should read", "reaches"
    ]
    is_internal = any(ind in text_lower for ind in internal_indicators)

    # Numeric temperature (e.g., "375°F", "165 degrees F")
    temp_match = re.search(
        r"(\d{2,3})\s*(?:°|degrees?\s*)?"
        r"(F|C|Fahrenheit|Celsius)\b",
        text_stripped,
        flags=re.I,
    )
    
    if temp_match:
        temp_str = temp_match.group(0).strip()
        if is_internal:
            info["internal"] = temp_str
        else:
            info["cooking"] = temp_str  # covers oven, grill, etc.
    
    # Stovetop heat levels (e.g., "medium heat", "medium-high heat")
    heat_match = re.search(
        r"\b("
        r"low|"
        r"medium(?:-|\s)low|"
        r"medium(?:-|\s)high|"
        r"medium|"
        r"high"
        r")\s+heat\b",
        text_lower,
    )
    
    if heat_match:
        level = heat_match.group(1)
        level = level.replace(" ", "-")
        info["cooking"] = level + " heat"

    return info


def _pos_tag(text: str):
    if _nltk_word_tokenize is None or _nltk_pos_tag is None:
        raise RuntimeError(
            "NLTK is required for POS tagging but is not available. "
            "Please install nltk and run:\n"
            '    python -c "import nltk; nltk.download(\'punkt\'); nltk.download(\'averaged_perceptron_tagger\')"'
        )
    try:
        tokens = _nltk_word_tokenize(text)
        return _nltk_pos_tag(tokens)
    except LookupError as e:
        raise RuntimeError(
            "NLTK data for POS tagging is missing. Please run:\n"
            '    python -c "import nltk; nltk.download(\'punkt\'); nltk.download(\'averaged_perceptron_tagger\')"'
        ) from e


def extract_cooking_methods(text: str) -> List[str]:
    tagged = _pos_tag(text)
    methods: List[str] = []

    n = len(tagged)
    for i, (token, tag) in enumerate(tagged):
        is_verb_like = tag.startswith("VB")
        is_noun_like = tag in ("NN", "NNP")

        if not (is_verb_like or is_noun_like):
            continue
        if tag == "VBN" and i + 1 < n:
            next_tag = tagged[i + 1][1]
            if next_tag.startswith("NN"):
                continue
        if is_noun_like and i != 0:
            continue
        lemma = _lemmatize(token, "v")
        
        if lemma in COOKING_VERBS:
            methods.append(lemma)
    
    text_lower = text.lower()
    for verb in PRIMARY_METHODS + OTHER_METHODS:
        if " " in verb and verb in text_lower:
            methods.append(verb)
    seen = set()
    unique: List[str] = []
    
    for m in methods:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    
    return unique


def extract_tools_from_text(text: str) -> List[str]:
    tagged = _pos_tag(text)
    tools: List[str] = []
    for token, tag in tagged:
        if tag.startswith("NN"):
            lemma = _lemmatize(token, "n")
            canonical = TOOL_LEMMA_TO_CANONICAL.get(lemma)
            if canonical:
                tools.append(canonical)

    multiword_tools = [t for t in TOOLS if " " in t]
    tools.extend(find_items_in_text(text, multiword_tools))
    seen = set()
    unique: List[str] = []
    for t in tools:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique




# FIX #6: Improved ingredient matching with lemmatization for singular/plural
def ingredient_matches_step(ingredient_name: str, text_lower: str) -> bool:
    name = ingredient_name.lower().strip()
    if not name:
        return False
    if re.search(r"\b" + re.escape(name) + r"\b", text_lower):
        return True
    words = re.findall(r"[a-zA-Z]+", name)
    if not words:
        return False

    IGNORE_WORDS = {"of", "and", "or", "in", "with", "to", "for", "the", "a", "an"}
    for w in words:
        if w in IGNORE_WORDS or w in DESCRIPTOR_WORDS:
            continue
        # Check exact word match
        if re.search(r"\b" + re.escape(w) + r"\b", text_lower):
            return True
        # Check lemmatized/singular form (e.g., "eggs" -> "egg")
        lemma = _lemmatize(w, "n")
        if lemma != w and re.search(r"\b" + re.escape(lemma) + r"\b", text_lower):
            return True
        # Check if ingredient is plural but text has singular
        if w.endswith("s") and len(w) > 2:
            singular = w[:-1]
            if re.search(r"\b" + re.escape(singular) + r"\b", text_lower):
                return True
    return False






def build_steps(steps_raw: List[str], ingredients: List[Ingredient]) -> List[Step]:
    steps: List[Step] = []
    ingredient_names = [ing.name.lower() for ing in ingredients]
    current_cooking_temp: Optional[str] = None
    step_counter = 1

    for raw_step in steps_raw:
        atomic_texts = split_into_atomic_steps(raw_step)
        for text in atomic_texts:
            text = text.strip()
            if not text:
                continue
            text_lower = text.lower()
            tools = extract_tools_from_text(text)
            methods = extract_cooking_methods(text)
            time_info = extract_time(text)
            temp_info = extract_temperature(text)
            used_ingredients: List[str] = []
            for name in ingredient_names:
                if ingredient_matches_step(name, text_lower):
                    used_ingredients.append(name)


            if "cooking" in temp_info:
                current_cooking_temp = temp_info["cooking"]

            action = methods[0] if methods else None
            context: Dict[str, str] = {}
            
            if current_cooking_temp:
                context["cooking_temperature"] = current_cooking_temp

            step = Step(
                step_number=step_counter,
                description=text,
                ingredients=used_ingredients,
                tools=tools,
                methods=methods,
                time=time_info,
                temperature=temp_info,
                action=action,
                objects=used_ingredients,
                modifiers={"tools": ", ".join(tools)} if tools else {},
                context=context,
            )
            steps.append(step)
            step_counter += 1

    return steps




def collect_recipe_tools_and_methods(steps: List[Step]) -> Tuple[List[str], List[str]]:
    tools_set = set()
    methods_set = set()
    for s in steps:
        tools_set.update(s.tools)
        methods_set.update(s.methods)
    return sorted(tools_set), sorted(methods_set)




# FIX #2: Check if first step already has "gather" before adding synthetic one
def parse_recipe_from_url(url: str) -> Recipe:
    html = fetch_html(url)
    base = parse_allrecipes_basic(html)

    ingredients = parse_ingredients(base["ingredients_raw"])
    steps = build_steps(base["steps_raw"], ingredients)
    ingredient_names_lower = [
        ing.name.lower() for ing in ingredients if ing.name
    ]

    # FIX #2: Only add gather step if first step doesn't already mention "gather"
    first_step_text = steps[0].description.lower() if steps else ""
    should_add_gather = "gather" not in first_step_text

    if should_add_gather:
        gather_step = Step(
            step_number=1,
            description="Gather all ingredients.",
            ingredients=ingredient_names_lower,
            tools=[],
            methods=["gather"],
            time={},
            temperature={},
            action="gather",
            objects=ingredient_names_lower,
            modifiers={},
            context={},
        )
        for idx, step in enumerate(steps, start=2):
            step.step_number = idx
        steps = [gather_step] + steps

    tools, methods = collect_recipe_tools_and_methods(steps)

    return Recipe(
        title=base["title"],
        url=url,
        ingredients=ingredients,
        tools=tools,
        methods=methods,
        steps=steps,)




# FIX #1: Output flat list of methods instead of primary/other split
def recipe_to_json(recipe: Recipe) -> Dict[str, object]:
    ingredients_json: List[Dict[str, object]] = []
    for ing in recipe.ingredients:
        ingredients_json.append(
            {
                "name": ing.name,
                "quantity": ing.quantity,
                "measurement": ing.unit,
                "descriptor": ing.descriptor,
                "preparation": ing.preparation,
            }
        )

    # FIX #1: Flat list of methods
    methods_json = list(recipe.methods)

    steps_json: List[Dict[str, object]] = []
    for step in recipe.steps:
        steps_json.append(
            {
                "step_number": step.step_number,
                "description": step.description,
                "ingredients": step.ingredients,
                "tools": step.tools,
                "methods": step.methods,  # Already a flat list
                "time": {
                    "duration": step.time.get("duration"),
                },
                "temperature": step.temperature,
            }
        )

    return {
        "title": recipe.title,
        "ingredients": ingredients_json,
        "tools": recipe.tools,
        "methods": methods_json,
        "steps": steps_json,
    }




if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Recipe URL to parse")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON format")
    parser.add_argument("--out", type=str, default=None,
                        help="Save output to file")
    args = parser.parse_args()
    recipe = parse_recipe_from_url(args.url)

    if args.json:
        data = json.dumps(recipe_to_json(recipe), indent=2, ensure_ascii=False)

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(data)
            print(f"JSON saved to {args.out}")
        else:
            print(data)
    else:
        print("Title:", recipe.title)
        print("Ingredients:", len(recipe.ingredients))
        print("Steps:", len(recipe.steps))

