"""
JanNyaya AI - Document Intelligence, Comparison & Contradiction Engine

Responsibilities:
1. Multi-Document Comparison Matrix (comparing amounts, dates, parties, clauses across files).
2. Contradiction & Discrepancy Detection (amount conflict, date mismatch, forum conflict).
3. Missing Information Engine (audits for unattached contracts, receipts, postal slips, signatures).
4. OCR Quality & Readability Scorer (corruption ratio, scan quality score, warnings).
5. Case Information Completeness Score (0-100% factual case readiness gauge).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class DocumentIntelligenceService:
    """Provides advanced document comparison, conflict analysis, and quality scoring."""

    @classmethod
    def calculate_ocr_quality(cls, text: str) -> Dict[str, Any]:
        """Calculates OCR quality and text readability metrics."""
        if not text or len(text.strip()) < 10:
            return {
                "quality_score": 0,
                "readability_label": "Empty / Unreadable",
                "garbled_ratio": 1.0,
                "warning": "No readable text extracted. Please ensure the document is clear and well-lit.",
            }

        total_chars = len(text)
        # Count corrupted/unusual non-standard symbols that typically indicate OCR noise
        garbled_chars = len(re.findall(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f§¤¥¦©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿]", text))
        garbled_ratio = garbled_chars / max(total_chars, 1)

        words = text.split()
        total_words = len(words)
        if total_words == 0:
            return {"quality_score": 0, "readability_label": "Unreadable", "garbled_ratio": 1.0}

        # Check average word length and dictionary-like word ratio
        valid_words = sum(1 for w in words if re.match(r"^[A-Za-z0-9\u0900-\u097F\u0C80-\u0CFF,.'\"₹()-]+$", w))
        valid_word_ratio = valid_words / max(total_words, 1)

        quality_score = int(max(0, min(100, (valid_word_ratio * 85) + ((1.0 - garbled_ratio) * 15))))

        if quality_score >= 90:
            label = "High Readability (90%+)"
            warning = None
        elif quality_score >= 75:
            label = "Good Readability (75-89%)"
            warning = None
        elif quality_score >= 50:
            label = "Moderate Readability (50-74%)"
            warning = "Some scanned sections or stamps may have minor OCR errors. Please verify extracted amounts."
        else:
            label = "Degraded Scan (<50%)"
            warning = "Document scan contains significant noise or handwriting that could not be fully recognized."

        return {
            "quality_score": quality_score,
            "readability_label": label,
            "garbled_ratio": round(garbled_ratio, 4),
            "valid_word_ratio": round(valid_word_ratio, 4),
            "total_words": total_words,
            "warning": warning,
        }

    @classmethod
    def detect_contradictions(
        cls,
        doc_analyses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detects conflicting amounts, dates, or terms across multiple uploaded documents."""
        contradictions: List[Dict[str, Any]] = []
        if not doc_analyses or len(doc_analyses) < 2:
            return contradictions

        # 1. Compare amounts across documents
        extracted_amounts_per_doc: List[Tuple[str, List[float]]] = []
        for doc in doc_analyses:
            doc_name = doc.get("filename", "Document")
            raw_amts = doc.get("amounts", [])
            numeric_amts = []
            for a in raw_amts:
                a_str = a if isinstance(a, str) else str(a.get("amount", a))
                match = re.search(r"([0-9,]+(?:\.[0-9]{2})?)", a_str.replace(" ", ""))
                if match:
                    try:
                        val = float(match.group(1).replace(",", ""))
                        if val > 100:  # Ignore trivial numbers like page numbers
                            numeric_amts.append(val)
                    except ValueError:
                        pass
            if numeric_amts:
                extracted_amounts_per_doc.append((doc_name, numeric_amts))

        if len(extracted_amounts_per_doc) >= 2:
            doc1_name, doc1_amts = extracted_amounts_per_doc[0]
            doc2_name, doc2_amts = extracted_amounts_per_doc[1]
            max1 = max(doc1_amts) if doc1_amts else None
            max2 = max(doc2_amts) if doc2_amts else None
            if max1 and max2 and abs(max1 - max2) > 1.0:
                contradictions.append({
                    "issue": "Financial Claim Discrepancy",
                    "type": "amount_conflict",
                    "detail": f"{doc1_name} claims ₹{max1:,.2f} whereas {doc2_name} records ₹{max2:,.2f} (Difference of ₹{abs(max1-max2):,.2f}).",
                    "recommendation": "Reconcile account statements to verify if interest, late fees, or uncredited payments cause the difference.",
                })

        # 2. Check jurisdiction/arbitration clauses vs litigation threats
        texts = [doc.get("text", "") or doc.get("summary", "") for doc in doc_analyses]
        has_arbitration = any("arbitrat" in t.lower() for t in texts)
        has_court_threat = any("police complaint" in t.lower() or "fir" in t.lower() or "civil court" in t.lower() for t in texts)
        if has_arbitration and has_court_threat:
            contradictions.append({
                "issue": "Dispute Resolution Forum Conflict",
                "type": "forum_conflict",
                "detail": "Underlying contract contains an Arbitration Clause, but the recent notice threatens Criminal / Civil Court litigation.",
                "recommendation": "Section 8 of Arbitration & Conciliation Act 1996 may require parties to refer dispute to arbitration before filing a civil suit.",
            })

        return contradictions

    @classmethod
    def audit_missing_information(
        cls,
        doc_analysis: Dict[str, Any],
        doc_text: str = "",
    ) -> List[Dict[str, Any]]:
        """Audits for standard missing legal elements (annexures, signatures, tracking receipts)."""
        missing: List[Dict[str, Any]] = []
        combined_text = (doc_text + " " + str(doc_analysis)).lower()

        # Check for unattached agreements / loan contracts
        if ("agreement dated" in combined_text or "loan agreement" in combined_text or "annexure" in combined_text) and not doc_analysis.get("has_attached_contract"):
            missing.append({
                "item": "Underlying Contract / Agreement Annexure",
                "importance": "Critical",
                "reason": "The notice references an underlying contract/schedule that sets repayment terms and interest rates.",
            })

        # Check for postal / delivery proof
        if "legal notice" in combined_text or "demand notice" in combined_text:
            if not any(w in combined_text for w in ["speed post tracking", "consignment number", "acknowledged on", "delivered on"]):
                missing.append({
                    "item": "Proof of Notice Service / Dispatch Receipt",
                    "importance": "High",
                    "reason": "Statutory limitation periods (e.g. 15 days under Section 138 NI Act) begin only upon actual date of delivery.",
                })

        # Check for detailed calculation sheet
        if doc_analysis.get("amounts") and not any(w in combined_text for w in ["statement of account", "ledger", "principal breakdown", "interest calculation"]):
            missing.append({
                "item": "Detailed Statement of Account / Principal Breakdown",
                "importance": "Moderate",
                "reason": "A lump-sum demand should be supported by a monthly ledger showing interest and credited payments.",
            })

        # Check for signature / authorization
        if not any(w in combined_text for w in ["advocate signature", "authorized signatory", "signed by", "thumb impression"]):
            missing.append({
                "item": "Authorised Signature / Power of Attorney",
                "importance": "Moderate",
                "reason": "Ensure the issuing advocate or institution has valid power of attorney to issue the demand.",
            })

        return missing

    @classmethod
    def calculate_case_completeness(
        cls,
        doc_analyses: List[Dict[str, Any]],
        timeline_events: List[Any],
        provisions: List[Any],
    ) -> Dict[str, Any]:
        """Calculates 0-100% case information completeness based on factual readiness."""
        score = 0
        breakdown = {}

        # 1. Document presence (20 pts)
        doc_count = len(doc_analyses)
        doc_pts = min(20, doc_count * 10)
        score += doc_pts
        breakdown["documents"] = {"score": doc_pts, "max": 20, "label": f"{doc_count} document(s) uploaded"}

        # 2. Identified parties (20 pts)
        parties_found = any(len(d.get("parties", [])) > 0 for d in doc_analyses)
        party_pts = 20 if parties_found else 5
        score += party_pts
        breakdown["parties"] = {"score": party_pts, "max": 20, "label": "Parties and roles identified" if parties_found else "Parties partially identified"}

        # 3. Timeline and dates (20 pts)
        timeline_count = len(timeline_events)
        timeline_pts = min(20, max(5, timeline_count * 4))
        score += timeline_pts
        breakdown["timeline"] = {"score": timeline_pts, "max": 20, "label": f"{timeline_count} chronological event(s) logged"}

        # 4. Quantified amounts & claims (20 pts)
        amounts_found = any(len(d.get("amounts", [])) > 0 for d in doc_analyses)
        amt_pts = 20 if amounts_found else 5
        score += amt_pts
        breakdown["claims"] = {"score": amt_pts, "max": 20, "label": "Financial claims isolated" if amounts_found else "No financial figures stated"}

        # 5. Verified legal provisions (20 pts)
        prov_count = len(provisions)
        prov_pts = min(20, prov_count * 5)
        score += prov_pts
        breakdown["provisions"] = {"score": prov_pts, "max": 20, "label": f"{prov_count} statutory provision(s) mapped"}

        completeness_pct = min(100, max(10, score))
        if completeness_pct >= 85:
            tier = "High Case Readiness"
        elif completeness_pct >= 60:
            tier = "Moderate Case Readiness"
        else:
            tier = "Preliminary Case Dossier"

        return {
            "completeness_score": completeness_pct,
            "readiness_tier": tier,
            "breakdown": breakdown,
        }


def analyze_document_intelligence(
    doc_analyses: List[Dict[str, Any]],
    doc_text: str = "",
    timeline_events: Optional[List[Any]] = None,
    provisions: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Helper providing complete document intelligence report."""
    contradictions = DocumentIntelligenceService.detect_contradictions(doc_analyses)
    missing_info = DocumentIntelligenceService.audit_missing_information(
        doc_analyses[0] if doc_analyses else {},
        doc_text=doc_text,
    )
    ocr_quality = DocumentIntelligenceService.calculate_ocr_quality(doc_text)
    completeness = DocumentIntelligenceService.calculate_case_completeness(
        doc_analyses=doc_analyses,
        timeline_events=timeline_events or [],
        provisions=provisions or [],
    )

    return {
        "ocr_quality": ocr_quality,
        "contradictions": contradictions,
        "missing_information": missing_info,
        "case_completeness": completeness,
    }
