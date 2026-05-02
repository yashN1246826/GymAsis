import re
import nltk
from google import genai
from nltk.sem.logic import Expression
from nltk.inference import ResolutionProver

nltk.download('punkt', quiet=True)

read_expr = Expression.fromstring

PROJECT_ID = "project-ddff7863-446f-4384-b6b"
LOCATION = "global"
MODEL_NAME = "gemini-2.5-flash"


class LogicEngine:
    """
    First-order logic knowledge base and inference engine for GymBot.
    """

    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.kb_expressions = []
        self.kb_raw_strings = []
        self._load_kb(kb_path)
        print(f"[LogicEngine] Loaded {len(self.kb_expressions)} KB expressions.")
        self._check_contradictions()

    def _load_kb(self, path: str):
        self.kb_expressions = []
        self.kb_raw_strings = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    expr = read_expr(line)
                    self.kb_expressions.append(expr)
                    self.kb_raw_strings.append(line)
                except Exception as e:
                    print(f"[LogicEngine] WARNING: Could not parse '{line}': {e}")

    def _check_contradictions(self):
        prover = ResolutionProver()
        try:
            result = prover.prove(
                read_expr('False'),
                self.kb_expressions,
                verbose=False
            )
            if result:
                print("[LogicEngine] WARNING: Knowledge base contains a contradiction!")
            else:
                print("[LogicEngine] Knowledge base integrity check passed — no contradictions.")
        except Exception as e:
            print(f"[LogicEngine] Contradiction check could not complete: {e}")

    def _normalise_term(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r'^(the|a|an)\s+', '', text, flags=re.IGNORECASE)
        text = text.replace(' ', '_')
        return text

    def _capitalise_predicate(self, predicate_text: str) -> str:
        predicate_text = self._normalise_term(predicate_text)
        return predicate_text[0].upper() + predicate_text[1:] if predicate_text else predicate_text

    def _parse_x_is_y(self, sentence: str):
        sentence = re.sub(r'^(the|a|an)\s+', '', sentence.strip(), flags=re.IGNORECASE)

        match = re.match(r'^(.+?)\s+is\s+(a\s+|an\s+|the\s+)?(.+)$',
                         sentence.strip(), re.IGNORECASE)
        if not match:
            return None, None, None

        subject = self._normalise_term(match.group(1))
        predicate = self._capitalise_predicate(match.group(3))
        fol_string = f"{predicate}({subject})"
        return subject, predicate, fol_string

    def _parse_x_trains_y(self, sentence: str):
        match = re.match(r'^(.+?)\s+trains\s+(.+)$', sentence.strip(), re.IGNORECASE)
        if not match:
            return None, None, None

        subject = self._normalise_term(match.group(1))
        obj = self._normalise_term(match.group(2))
        fol_string = f"Trains({subject},{obj})"
        return subject, obj, fol_string

    def _validate_fact_with_api(self, user_statement: str, statement_type: str) -> tuple[bool, str]:
        """
        Uses Google Gemini to check whether a new fact is gym-related and plausible.
        Returns (is_valid, explanation).
        """
        try:
            client = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location=LOCATION,
            )

            if statement_type == "is":
                task_text = (
                    "The user wants to add a gym knowledge fact in the form 'X is Y'. "
                    "Decide if the fact is gym-related and plausible."
                )
            else:
                task_text = (
                    "The user wants to add a gym knowledge fact in the form 'X trains Y'. "
                    "Decide if the fact is gym-related and plausible."
                )

            prompt = f"""
You are validating facts for a gym training chatbot.

{task_text}

Return your answer in exactly one of these formats only:
VALID: short reason
INVALID: short reason

Fact:
{user_statement}
"""

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            text = getattr(response, "text", None)
            if not text:
                return True, "Validation unavailable, using local checks only."

            cleaned = text.strip()

            if cleaned.upper().startswith("VALID:"):
                return True, cleaned[6:].strip()

            if cleaned.upper().startswith("INVALID:"):
                return False, cleaned[8:].strip()

            return True, "Validation unclear, using local checks only."

        except Exception as e:
            print(f"[LogicEngine] API validation error: {e}")
            return True, "Validation unavailable, using local checks only."

    def handle_i_know(self, x: str, y: str) -> str:
        subject, predicate, fol_string = self._parse_x_is_y(f"{x} is {y}")

        if fol_string is None:
            return (f"Sorry, I could not parse '{x} is {y}'. "
                    "Please use the format: I know that X is Y.")

        try:
            new_expr = read_expr(fol_string)
        except Exception as e:
            return f"I could not understand '{fol_string}' as a logical expression: {e}"

        if fol_string in self.kb_raw_strings:
            return f"I already know that {x} is {y}."

        api_ok, api_reason = self._validate_fact_with_api(f"{x} is {y}", "is")
        if not api_ok:
            return (
                f"Sorry, I cannot add that because the Google API validation flagged it as doubtful. "
                f"Reason: {api_reason}"
            )

        negated = read_expr(f"-{fol_string}")
        prover = ResolutionProver()
        try:
            is_contradicted = prover.prove(negated, self.kb_expressions, verbose=False)
        except Exception:
            is_contradicted = False

        if is_contradicted:
            return f"Sorry, this contradicts with what I know! I cannot add that {x} is {y}."

        self.kb_expressions.append(new_expr)
        self.kb_raw_strings.append(fol_string)
        return f"OK, I will remember that {x} is {y}."

    def handle_check(self, x: str, y: str) -> str:
        subject, predicate, fol_string = self._parse_x_is_y(f"{x} is {y}")

        if fol_string is None:
            return (f"Sorry, I could not parse '{x} is {y}'. "
                    "Please use the format: Check that X is Y.")

        try:
            goal_expr = read_expr(fol_string)
            negated_expr = read_expr(f"-{fol_string}")
        except Exception as e:
            return f"Could not parse the expression: {e}"

        prover = ResolutionProver()

        try:
            positive = prover.prove(goal_expr, self.kb_expressions, verbose=False)
        except Exception:
            positive = False

        if positive:
            return f"Correct. Based on my knowledge base, {x} is {y}."

        try:
            negative = prover.prove(negated_expr, self.kb_expressions, verbose=False)
        except Exception:
            negative = False

        if negative:
            return (f"It may not be true... let me check...\n"
                    f"Incorrect. Based on my knowledge base, {x} is NOT {y}.")

        return (f"It may not be true... let me check...\n"
                f"Sorry, I don't know whether {x} is {y}. "
                f"It is not in my knowledge base.")

    def handle_i_know_trains(self, x: str, y: str) -> str:
        subject, obj, fol_string = self._parse_x_trains_y(f"{x} trains {y}")

        if fol_string is None:
            return (f"Sorry, I could not parse '{x} trains {y}'. "
                    "Please use the format: I know that X trains Y.")

        try:
            new_expr = read_expr(fol_string)
        except Exception as e:
            return f"I could not understand '{fol_string}' as a logical expression: {e}"

        if fol_string in self.kb_raw_strings:
            return f"I already know that {x} trains {y}."

        api_ok, api_reason = self._validate_fact_with_api(f"{x} trains {y}", "trains")
        if not api_ok:
            return (
                f"Sorry, I cannot add that because the Google API validation flagged it as doubtful. "
                f"Reason: {api_reason}"
            )

        negated = read_expr(f"-{fol_string}")
        prover = ResolutionProver()
        try:
            is_contradicted = prover.prove(negated, self.kb_expressions, verbose=False)
        except Exception:
            is_contradicted = False

        if is_contradicted:
            return f"Sorry, this contradicts with what I know! I cannot add that {x} trains {y}."

        self.kb_expressions.append(new_expr)
        self.kb_raw_strings.append(fol_string)
        return f"OK, I will remember that {x} trains {y}."

    def handle_check_trains(self, x: str, y: str) -> str:
        subject, obj, fol_string = self._parse_x_trains_y(f"{x} trains {y}")

        if fol_string is None:
            return (f"Sorry, I could not parse '{x} trains {y}'. "
                    "Please use the format: Check that X trains Y.")

        try:
            goal_expr = read_expr(fol_string)
            negated_expr = read_expr(f"-{fol_string}")
        except Exception as e:
            return f"Could not parse the expression: {e}"

        prover = ResolutionProver()

        try:
            positive = prover.prove(goal_expr, self.kb_expressions, verbose=False)
        except Exception:
            positive = False

        if positive:
            return f"Correct. Based on my knowledge base, {x} trains {y}."

        try:
            negative = prover.prove(negated_expr, self.kb_expressions, verbose=False)
        except Exception:
            negative = False

        if negative:
            return (f"It may not be true... let me check...\n"
                    f"Incorrect. Based on my knowledge base, {x} does NOT train {y}.")

        return (f"It may not be true... let me check...\n"
                f"Sorry, I don't know whether {x} trains {y}. "
                f"It is not in my knowledge base.")

    def display_kb(self):
        print("\n--- Current Knowledge Base ---")
        for i, expr_str in enumerate(self.kb_raw_strings, 1):
            print(f"  {i:02d}. {expr_str}")
        print(f"  Total: {len(self.kb_raw_strings)} expressions")
        print("------------------------------\n")