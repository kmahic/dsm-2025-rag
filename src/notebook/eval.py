"""
Evaluering av RAG-svar med LLM-as-a-Judge (Gemini).

Inneholder metoder for å vurdere:
- Relevans: Hvor godt svarer svaret på spørsmålet?
- Groundedness: Er svaret forankret i kildene?
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from google import genai
from google.genai.types import GenerateContentConfig
from google import genai as genai_module

# Standard modell for evaluering
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"


def _extract_text_or_none(resp) -> Optional[str]:
    """Hent første tekstpart fra første kandidat. Returner None ved truncation/ingen parts."""
    try:
        cand = resp.candidates[0]
        fr = getattr(cand, "finish_reason", None)
        if fr and str(fr).endswith("MAX_TOKENS"):
            return None
        parts = getattr(cand, "content", None) and cand.content.parts
        if not parts:
            return None
        return parts[0].text.strip()
    except Exception:
        return None


def _ask_gemini_score(
    prompt: str,
    min_val: int = 0,
    max_val: int = 5,
    model: str = DEFAULT_JUDGE_MODEL,
    client: Optional[genai.Client] = None,  # type: ignore
) -> Optional[float]:
    """Kaller Gemini og forventer en score (0..5). Returnerer None ved feil/manglende tall."""
    if client is None:
        if genai_module is None:
            raise ImportError("google-genai pakken er ikke installert")
        client = genai_module.Client()

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=64,
                candidate_count=1,
                top_k=1,
            ),
        )
        text = _extract_text_or_none(resp)
        if not text:
            return None

        # Prøv å ekstrahere tall fra responsen
        for tok in text.replace(",", ".").split():
            try:
                val = float(tok)
                return max(min_val, min(max_val, val))
            except ValueError:
                continue
        return None
    except Exception as e:
        print(f"⚠️  Feil ved kall til Gemini: {e}")
        return None


def relevance_to_query_score(
    question: str,
    answer: str,
    model: str = DEFAULT_JUDGE_MODEL,
    client: Optional[genai.Client] = None,  # type: ignore
) -> Optional[float]:
    """
    Evaluerer hvor relevant svaret er for spørsmålet.

    Args:
        question: Brukerens spørsmål
        answer: RAG-systemets svar
        model: Gemini-modell for evaluering
        client: Optional Gemini-klient

    Returns:
        Score 0-5, hvor:
        - 0 = helt irrelevant
        - 3 = delvis relevant men ufullstendig
        - 5 = direkte og presist svar
        - None = feil ved evaluering
    """
    if not question or not answer:
        return None

    prompt = (
        "Gi en score (kun tallet) for hvor relevant SVAR er for SPØRSMÅLET, 0 til 5.\n"
        "0 = helt irrelevant, 3 = delvis relevant men ufullstendig, 5 = direkte og presist svar.\n\n"
        f"SPØRSMÅL:\n{question}\n\nSVAR:\n{answer}\n\n"
        "Svar KUN med ett tall mellom 0 og 5 på første linje. (kun et tall):"
    )
    return _ask_gemini_score(prompt, 0, 5, model=model, client=client)


def retrieval_groundedness_score(
    answer: str,
    sources: List[Dict[str, Any]],
    model: str = DEFAULT_JUDGE_MODEL,
    client: Optional[genai.Client] = None,  # type: ignore
) -> Optional[float]:
    """
    Evaluerer om svaret er forankret i de oppgitte kildene.

    Args:
        answer: RAG-systemets svar
        sources: Liste med kildeinfo, hver dict skal ha:
            - text_preview/text: tekst fra kilden
            - retrieval_path (optional): sti til chunk
            - source (optional): dokumentnavn
        model: Gemini-modell for evaluering
        client: Optional Gemini-klient

    Returns:
        Score 0-5, hvor:
        - 0 = ikke forankret i kildene
        - 3 = delvis forankret
        - 5 = tydelig forankret i kildene
        - None = feil ved evaluering
    """
    if not answer or not sources:
        return None

    # Bygg kontekst-blokk fra kilder
    snippets: list[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        path = s.get("retrieval_path", "")
        text_prev = s.get("text_preview") or s.get("text", "")
        source = s.get("source", "")

        snippet_parts = []
        if path:
            snippet_parts.append(f"STI: {path}")
        if source:
            snippet_parts.append(f"KILDE: {source}")
        if text_prev:
            snippet_parts.append(f"UTDRAG: {text_prev}")

        if snippet_parts:
            snippets.append(" | ".join(snippet_parts))

    context_block = "\n\n".join(snippets) if snippets else "(ingen utdrag)"

    prompt = (
        "Gitt KONTEKST (utdrag fra dokumenter) og SVAR, gi en forankringsscore 0..5 "
        "for hvor godt påstandene kan verifiseres i konteksten.\n"
        "0 = ikke forankret, 3 = delvis forankret, 5 = tydelig forankret.\n\n"
        f"KONTEKST:\n{context_block}\n\nSVAR:\n{answer}\n\n"
        "Svar KUN med ett tall mellom 0 og 5 på første linje. (kun et tall):"
    )
    return _ask_gemini_score(prompt, 0, 5, model=model, client=client)


def evaluate_rag_response(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    model: str = DEFAULT_JUDGE_MODEL,
    client: Optional[genai.Client] = None,  # type: ignore
    threshold: float = 4.0,
) -> Dict[str, Any]:
    """
    Kjører full evaluering av et RAG-svar.

    Args:
        question: Brukerens spørsmål
        answer: RAG-systemets svar
        sources: Liste med kildeinfo
        model: Gemini-modell for evaluering
        client: Optional Gemini-klient
        threshold: Terskelverdi for pass (standard 4.0)

    Returns:
        Dict med evalueringsresultater:
        - relevance_score: float eller None
        - groundedness_score: float eller None
        - pass_relevance: bool
        - pass_groundedness: bool
        - pass_all: bool
    """
    relevance = relevance_to_query_score(
        question, answer, model=model, client=client)
    groundedness = retrieval_groundedness_score(
        answer, sources, model=model, client=client)

    pass_relevance = relevance is not None and relevance >= threshold
    pass_groundedness = groundedness is not None and groundedness >= threshold
    pass_all = pass_relevance and pass_groundedness

    return {
        "relevance_score": relevance,
        "groundedness_score": groundedness,
        "pass_relevance": pass_relevance,
        "pass_groundedness": pass_groundedness,
        "pass_all": pass_all,
    }


def print_evaluation_results(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    model: str = DEFAULT_JUDGE_MODEL,
    client: Optional[genai.Client] = None,  # type: ignore
    threshold: float = 4.0,
) -> Dict[str, Any]:
    """
    Evaluerer RAG-svar og printer resultatet.

    Eksempel:
        >>> sources = [
        ...     {"text": "Dokumentavgift er 2,5%", "source": "dokumentavgiftsloven.pdf"}
        ... ]
        >>> results = print_evaluation_results(
        ...     question="Hva er dokumentavgiften?",
        ...     answer="Dokumentavgiften er 2,5% av kjøpesummen.",
        ...     sources=sources
        ... )
    """
    print("\n" + "="*60)
    print("📊 RAG EVALUERING")
    print("="*60)
    print(f"\n❓ Spørsmål: {question}")
    print(f"\n💬 Svar: {answer}")
    print(f"\n📚 Antall kilder: {len(sources)}")

    results = evaluate_rag_response(
        question=question,
        answer=answer,
        sources=sources,
        model=model,
        client=client,
        threshold=threshold,
    )

    print("\n" + "-"*60)
    print("📈 RESULTATER:")
    print("-"*60)

    # Relevans
    rel_score = results["relevance_score"]
    if rel_score is not None:
        rel_emoji = "✅" if results["pass_relevance"] else "❌"
        print(f"{rel_emoji} Relevans: {rel_score:.1f}/5.0 (terskel: {threshold})")
    else:
        print("⚠️  Relevans: Ingen score (dommer-feil)")

    # Groundedness
    grd_score = results["groundedness_score"]
    if grd_score is not None:
        grd_emoji = "✅" if results["pass_groundedness"] else "❌"
        print(f"{grd_emoji} Forankring: {grd_score:.1f}/5.0 (terskel: {threshold})")
    else:
        print("⚠️  Forankring: Ingen score (dommer-feil)")

    # Samlet resultat
    print("-"*60)
    if results["pass_all"]:
        print("🎉 BESTÅTT: Svaret er både relevant og godt forankret!")
    else:
        print("⚠️  IKKE BESTÅTT")
        if not results["pass_relevance"]:
            print("   - Svaret er ikke relevant nok for spørsmålet")
        if not results["pass_groundedness"]:
            print("   - Svaret er ikke godt nok forankret i kildene")

    print("="*60 + "\n")

    return results


# Eksempel på bruk
if __name__ == "__main__":
    # Eksempel 1: Godt svar
    print("EKSEMPEL 1: Godt svar")
    sources1 = [
        {
            "text": "Dokumentavgift er en avgift som skal betales ved tinglysing av dokumenter "
                    "som overfører hjemmel til fast eiendom. Avgiftssatsen er 2,5 prosent.",
            "source": "dokumentavgiftsloven.pdf",
            "retrieval_path": "chunk_42"
        }
    ]

    _ = print_evaluation_results(
        question="Hva er dokumentavgiften?",
        answer="Dokumentavgift er en avgift på 2,5% som betales ved tinglysing av eiendomsoverdragelser.",
        sources=sources1,
    )

    # Eksempel 2: Irrelevant svar
    print("\nEKSEMPEL 2: Irrelevant svar")
    _ = print_evaluation_results(
        question="Hva er dokumentavgiften?",
        answer="I Norge har vi et godt velferdssystem med høy skatt.",
        sources=sources1,
    )

    # Eksempel 3: Ikke forankret i kilder
    print("\nEKSEMPEL 3: Ikke forankret i kilder")
    sources3 = [
        {
            "text": "Eiendomsmegling reguleres av eiendomsmeglingsloven.",
            "source": "eiendomsmeglingsloven.pdf"
        }
    ]

    _ = print_evaluation_results(
        question="Hva er dokumentavgiften?",
        answer="Dokumentavgiften er 2,5% av kjøpesummen.",
        sources=sources3,
    )
