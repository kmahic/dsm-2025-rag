"""
Query rewriting strategies for RAG systems.

Provides different approaches to reformulate user questions for better retrieval:
- Multi-angle: Generate alternative phrasings with different perspectives
- Hypothetical answer: Create a hypothetical answer to match against documents
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from google import genai

try:
    from google.genai.types import GenerateContentConfig
    from google import genai as genai_module
except ImportError:
    GenerateContentConfig = None  # type: ignore
    genai_module = None  # type: ignore


DEFAULT_MODEL = "gemini-2.5-flash"


class QueryRewriter:
    """Base class for query rewriting strategies."""

    def __init__(
        self,
        client: genai.Client | None = None,  # type: ignore
        model: str = DEFAULT_MODEL,
    ):
        """
        Args:
            client: Gemini client (optional, will create if not provided)
            model: Model name for generation
        """
        if client is None:
            if genai_module is None:
                raise ImportError("google-genai pakken er ikke installert")
            client = genai_module.Client()

        self.client = client
        self.model = model

    def rewrite(self, user_input: str) -> List[str]:
        """
        Rewrite the user's question into one or more alternative queries.

        Args:
            user_input: Original user question

        Returns:
            List of rewritten queries
        """
        raise NotImplementedError("Must implement `rewrite` method")

    def _generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        """Helper to generate text from Gemini."""
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=512,
                    candidate_count=1,
                ),
            )

            if not resp.candidates:
                return ""

            parts = resp.candidates[0].content.parts
            if not parts:
                return ""

            return parts[0].text.strip()

        except Exception as e:
            print(f"⚠️  Feil ved generering: {e}")
            return ""


class MultiAngleRewriter(QueryRewriter):
    """
    Generates 1-2 alternative questions with different perspectives.

    Example:
        >>> rewriter = MultiAngleRewriter()
        >>> queries = rewriter.rewrite("Hva er dokumentavgiften?")
        >>> # Returns: ["Hvor mye koster dokumentavgift?", "Hvilken sats gjelder for dokumentavgift?"]
    """

    def rewrite(self, user_input: str) -> List[str]:
        """Generate alternative phrasings of the question."""
        prompt = f"""
Jeg har fått følgende spørsmål:
Spørsmål: ```{user_input}```

Jeg trenger å generere én eller to alternative spørsmål (queryer) som kan brukes for å hente relevant kontekst fra en vektorbasert søkemotor i en RAG-applikasjon.

Gi meg en kort liste med maks 2 spørsmål (1–2) som dekker spørsmålet på ulike måter, men uten at de betyr nesten det samme. Hvert spørsmål skal ha tydelig forskjellig vinkel eller fokus.

Format: ["Spørsmål 1", "Spørsmål 2"]
"""

        response = self._generate_text(prompt, temperature=0.0)

        # Parse JSON response
        try:
            queries = json.loads(response)
            if isinstance(queries, list) and queries:
                return queries
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return original question
        return [user_input]


class HypotheticalRewriter(QueryRewriter):
    """
    Generates a hypothetical answer to the question (HyDE approach).

    The hypothetical answer can be used to find similar documents in the vector store,
    as answers are often more similar to each other than questions are.

    Example:
        >>> rewriter = HypotheticalRewriter()
        >>> answer = rewriter.rewrite("Hva er dokumentavgiften?")
        >>> # Returns: ["Dokumentavgift er en avgift på 2,5% som betales ved kjøp av eiendom..."]
    """

    def rewrite(self, user_input: str) -> List[str]:
        """Generate a hypothetical answer to match against documents."""
        prompt = f"""
Du er en ekspert på kunnskapsbaserte AI-systemer. Du har fått følgende spørsmål fra en bruker:

Spørsmål: ```{user_input}```

Skriv et veldig kort hypotetisk dokument eller avsnitt som kunne vært et svar på spørsmålet (maks 100 ord), selv om du ikke har tilgang til noe konkret kontekst. Målet er å skape en representativ tekst som fanger spørsmålets intensjon, slik at denne teksten kan brukes til å hente relevant informasjon fra en vektor-database.

Skriv kun selve svaret. Ikke inkluder noen forklaring eller overskrift.
"""

        response = self._generate_text(prompt, temperature=0.3)

        if response:
            return [response]

        # Fallback: return original question
        return [user_input]


class StepBackRewriter(QueryRewriter):
    """
    Generates a more general "step-back" question.

    Useful for finding broader context before answering specific questions.

    Example:
        >>> rewriter = StepBackRewriter()
        >>> queries = rewriter.rewrite("Hva er dokumentavgiften på fritidsbolig?")
        >>> # Returns: ["Hva er dokumentavgift generelt?", "Hva er dokumentavgiften på fritidsbolig?"]
    """

    def rewrite(self, user_input: str) -> List[str]:
        """Generate a broader, more general version of the question."""
        prompt = f"""
Du er ekspert på å lage gode spørsmål for søk i dokumenter.

Brukeren har stilt dette spesifikke spørsmålet:
```{user_input}```

Lag ett mer generelt "step-back" spørsmål som dekker det grunnleggende temaet, før man går inn på detaljene. Dette spørsmålet skal hjelpe til med å finne bred bakgrunnskunnskap i dokumentbasen.

Svar med JSON format:
{{"general_question": "Det mer generelle spørsmålet", "specific_question": "{user_input}"}}

Eksempel:
Spesifikt: "Hva er dokumentavgiften på fritidsbolig?"
Generelt: "Hva er dokumentavgift?"
"""

        response = self._generate_text(prompt, temperature=0.0)

        # Parse JSON response
        try:
            data = json.loads(response)
            general = data.get("general_question", "").strip()
            specific = data.get("specific_question", user_input).strip()

            if general and general != specific:
                return [general, specific]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return original question
        return [user_input]


# Convenience function
def rewrite_query(
    user_input: str,
    strategy: str = "multi_angle",
    client: genai.Client | None = None,  # type: ignore
    model: str = DEFAULT_MODEL,
) -> List[str]:
    """
    Convenience function to rewrite a query using a specific strategy.

    Args:
        user_input: Original user question
        strategy: One of "multi_angle", "hypothetical", "step_back"
        client: Optional Gemini client
        model: Model name for generation

    Returns:
        List of rewritten queries

    Example:
        >>> queries = rewrite_query("Hva er dokumentavgiften?", strategy="multi_angle")
        >>> print(queries)
        ["Hvor mye koster dokumentavgift?", "Hvilken sats gjelder for dokumentavgift?"]
    """
    strategies = {
        "multi_angle": MultiAngleRewriter,
        "hypothetical": HypotheticalRewriter,
        "step_back": StepBackRewriter,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy}. Must be one of {list(strategies.keys())}"
        )

    rewriter = strategies[strategy](client=client, model=model)
    return rewriter.rewrite(user_input)


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("QUERY REWRITING EXAMPLES")
    print("="*60)

    question = "Hva er dokumentavgiften?"

    # Multi-angle rewriting
    print("\n1. MULTI-ANGLE REWRITING:")
    print(f"   Original: {question}")
    multi_rewriter = MultiAngleRewriter()
    multi_queries = multi_rewriter.rewrite(question)
    for i, q in enumerate(multi_queries, 1):
        print(f"   Alt {i}: {q}")

    # Hypothetical answer
    print("\n2. HYPOTHETICAL ANSWER (HyDE):")
    print(f"   Original: {question}")
    hyde_rewriter = HypotheticalRewriter()
    hyde_queries = hyde_rewriter.rewrite(question)
    for i, q in enumerate(hyde_queries, 1):
        print(f"   Answer {i}: {q[:100]}...")

    # Step-back question
    print("\n3. STEP-BACK REWRITING:")
    detailed_question = "Hva er dokumentavgiften på fritidsboliger?"
    print(f"   Original: {detailed_question}")
    stepback_rewriter = StepBackRewriter()
    stepback_queries = stepback_rewriter.rewrite(detailed_question)
    for i, q in enumerate(stepback_queries, 1):
        print(f"   Query {i}: {q}")

    # Using convenience function
    print("\n4. USING CONVENIENCE FUNCTION:")
    print(f"   Original: {question}")
    queries = rewrite_query(question, strategy="multi_angle")
    for i, q in enumerate(queries, 1):
        print(f"   Query {i}: {q}")

    print("\n" + "="*60)
