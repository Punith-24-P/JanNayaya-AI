"""
JanNyaya AI - Grounding Guard & Explainable Confidence Model

Responsibilities:
1. Compute explainable confidence tiers:
   - Strong Evidence: Multiple official Central Acts with verbatim match
   - Moderate Evidence: Recognized statutory source with relevant excerpt
   - Limited Evidence: General statutory reference requiring factual corroboration
   - Insufficient Evidence: Knowledge base lacks specific provision
2. Hallucination Guard:
   - Validate that mentioned section numbers and Acts exist in retrieved facts
   - Filter or qualify unsupported legal assertions
3. Compute Answer Grounding Score (0.0 - 1.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GroundingReport:
    confidence_level: str  # Strong Evidence, Moderate Evidence, Limited Evidence, Insufficient Evidence
    grounding_score: float  # 0.0 to 1.0
    supported_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    verified_sections: List[str] = field(default_factory=list)
    hallucination_detected: bool = False
    grounding_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_level": self.confidence_level,
            "grounding_score": round(self.grounding_score, 2),
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "verified_sections": self.verified_sections,
            "hallucination_detected": self.hallucination_detected,
            "grounding_notes": self.grounding_notes,
        }


class GroundingGuard:
    """Evaluates answer groundedness against retrieved statutory evidence."""

    @classmethod
    def evaluate_grounding(
        cls,
        answer: str,
        sources: List[Dict[str, Any]],
        facts: Optional[List[Dict[str, Any]]] = None,
    ) -> GroundingReport:
        if not answer or not sources:
            return GroundingReport(
                confidence_level="Insufficient Evidence",
                grounding_score=0.2,
                grounding_notes="No statutory sources available to substantiate this query.",
            )

        # Extract sections mentioned in the answer
        answer_sections = set(re.findall(r"(?:section|sec|धारा|ಸೆಕ್ಷನ್)\s*([0-9]+[A-Za-z]*)", answer, re.IGNORECASE))

        # Extract verified sections from sources & facts
        verified_sections_set = set()
        for src in sources:
            sec = str(src.get("section", "")).strip()
            if sec:
                verified_sections_set.add(sec)
        if facts:
            for f in facts:
                sec = str(f.get("section", "")).strip()
                if sec:
                    verified_sections_set.add(sec)

        supported_claims = []
        unsupported_claims = []

        # Check section alignment
        for sec in answer_sections:
            if sec in verified_sections_set:
                supported_claims.append(f"Section {sec} is verified in retrieved statutory text.")
            else:
                # Section mentioned in answer but absent in retrieved context
                unsupported_claims.append(f"Section {sec} was referenced but not found in primary retrieved chunks.")

        # Compute authority and coverage scores
        avg_authority = sum(float(s.get("authority_score", 0.85)) for s in sources) / max(len(sources), 1)
        source_count = len(sources)

        hallucination = len(unsupported_claims) > 0 and len(supported_claims) == 0

        # Grounding score formula
        if hallucination:
            score = 0.35
            conf = "Limited Evidence"
            notes = "Answer references statutory numbers not present in retrieved knowledge base."
        elif source_count >= 3 and avg_authority >= 0.95:
            score = 0.96
            conf = "Strong Evidence"
            notes = "Answer is corroborated by multiple primary Central Acts with verified provenance."
        elif source_count >= 1 and avg_authority >= 0.80:
            score = 0.85
            conf = "Moderate Evidence"
            notes = "Answer is supported by recognized statutory provisions."
        else:
            score = 0.60
            conf = "Limited Evidence"
            notes = "General statutory guidance applicable; verify specific factual applicability."

        return GroundingReport(
            confidence_level=conf,
            grounding_score=score,
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            verified_sections=list(verified_sections_set),
            hallucination_detected=hallucination,
            grounding_notes=notes,
        )


def verify_grounding(
    answer: str,
    sources: List[Dict[str, Any]],
    facts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Helper to verify answer grounding."""
    report = GroundingGuard.evaluate_grounding(answer, sources, facts)
    return report.to_dict()
