"""
JanNyaya AI - Hybrid Legal Retriever

Responsibilities
----------------
1. Semantic retrieval using multilingual E5 embeddings.
2. BM25-style lexical retrieval.
3. Reciprocal Rank Fusion (RRF).
4. Legal-topic detection.
5. Legal-route detection.
6. Route-aware reranking.
7. Topic-aware reranking.
8. Section-level deduplication.
9. Direct section expansion.

Supported topics
----------------
    theft
    murder
    cheating
    loan_recovery
    contract_breach
    property_dispute
    employment_dispute

Supported routes
----------------
    criminal
    civil_contractual
    property
    employment
    general

Important
---------
This module ranks retrieved legal evidence.
It does NOT decide legal applicability or legal liability.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from backend.embedding_service import (
    create_embedding,
)

from backend.vector_store import (
    get_all_documents,
    search_by_embedding,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SEMANTIC_K = 30
DEFAULT_BM25_K = 30
DEFAULT_FINAL_K = 8

RRF_K = 60.0

SECTION_EXPANSION_K = 3

MAX_BM25_DOCS = 10000

TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?"
)

# Base score weights.
WEIGHT_RRF = 1.00
WEIGHT_BM25 = 0.25
WEIGHT_SEMANTIC = 0.35
WEIGHT_TOPIC = 1.00
WEIGHT_ROUTE = 1.00
WEIGHT_TITLE = 0.80
WEIGHT_DEFINITION = 0.50
WEIGHT_PUNISHMENT = 0.70

# Extra bonuses.
BONUS_DIRECT_TITLE = 5.0
BONUS_SECTION_MATCH = 3.0
BONUS_PUNISHMENT_PHRASE = 2.5
BONUS_DEFINITION_PHRASE = 2.0
BONUS_SECTION_EXPANSION = 0.15

# Penalties.
PENALTY_WRONG_ROUTE = 12.0
PENALTY_UNRELATED_TOPIC = 5.0
PENALTY_SPECIALIZED = 2.0


# ============================================================
# CACHE
# ============================================================

_DOCUMENT_CACHE: Optional[
    Tuple[List[str], List[dict]]
] = None


_BM25_CACHE: Optional[
    Dict[str, Any]
] = None


# ============================================================
# LEGAL TOPIC CONFIGURATION
# ============================================================

TOPIC_CONFIG: Dict[str, Dict[str, Any]] = {

    "theft": {
        "route": "criminal",
        "terms": [
            "theft",
            "stolen",
            "stole",
            "dishonestly took",
            "movable property",
            "without consent",
            "taking property",
            "ಕಳ್ಳತನ",
            "ಕಳವು",
            "ಕದ್ದ",
            "ಚೋರಿ",
            "ಕಳ್ಳತನದ ಶಿಕ್ಷೆ",
            "ಚೋರ",
            "चोरी",
            "चुराया",
            "चोरी की सजा",
            "चोरी का अपराध",
        ],
        "strong_terms": [
            "theft",
            "stolen",
            "BNS 303",
            "Section 303",
            "ಕಳ್ಳತನ",
            "ಚೋರಿ",
            "चोरी",
        ],
        "queries": [
            "theft punishment Bharatiya Nyaya Sanhita Section 303",
            "theft offence dishonestly taking movable property without consent",
            "ಕಳ್ಳತನ ಅಪರಾಧ ಮತ್ತು ಶಿಕ್ಷೆ ಬಿಎನ್‌ಎಸ್",
            "चोरी की परिभाषा और सजा बीएनएस",
        ],
    },

    "murder": {
        "route": "criminal",
        "terms": [
            "murder",
            "killed",
            "caused the death",
            "death of",
            "homicide",
            "intentional death",
            "ಕೊಲೆ",
            "ಹತ್ಯೆ",
            "ಕೊಲೆಯ ಶಿಕ್ಷೆ",
            "ಪ್ರಾಣಹಾನಿ",
            "ജീവಹಾನಿ",
            "हत्या",
            "कत्ल",
            "जान से मारना",
            "हत्या की सजा",
        ],
        "strong_terms": [
            "murder",
            "homicide",
            "BNS 101",
            "BNS 103",
            "ಕೊಲೆ",
            "ಹತ್ಯೆ",
            "हत्या",
        ],
        "queries": [
            "murder punishment Bharatiya Nyaya Sanhita Section 101 103",
            "culpable homicide amounting to murder death penalty life imprisonment",
            "ಕೊಲೆ ಅಪರಾಧ ಮತ್ತು ಜೀವಾವಧಿ ಶಿಕ್ಷೆ",
            "हत्या की सजा और कानूनी प्रावधान बीएनएस",
        ],
    },

    "cheating": {
        "route": "criminal",
        "terms": [
            "cheating",
            "deception",
            "deceived",
            "dishonestly induced",
            "fraud",
            "false representation",
            "bogus scheme",
            "ವಂಚನೆ",
            "ಮೋಸ",
            "ಸುಳ್ಳು ಭರವಸೆ",
            "ವಂಚನೆಗೆ ಶಿಕ್ಷೆ",
            "ಮೋಸಗಾರ",
            "धोखाधड़ी",
            "जालसाजी",
            "छल",
            "गबन",
            "ठगी",
            "धोखा देकर पैसे हड़पना",
        ],
        "strong_terms": [
            "cheating",
            "fraud",
            "BNS 318",
            "Section 318",
            "IPC 420",
            "ವಂಚನೆ",
            "ಮೋಸ",
            "धोखाधड़ी",
        ],
        "queries": [
            "cheating punishment Bharatiya Nyaya Sanhita Section 318 319",
            "fraudulent deception dishonest inducement property",
            "ವಂಚನೆ ಮತ್ತು ಮೋಸ ಅಪರಾಧಕ್ಕೆ ಶಿಕ್ಷೆ",
            "धोखाधड़ी और जालसाजी की कानूनी सजा धारा 318",
        ],
    },

    "loan_recovery": {
        "route": "civil_contractual",
        "terms": [
            "loan",
            "loan amount",
            "loan agreement",
            "borrower",
            "lender",
            "repay",
            "repayment",
            "outstanding",
            "outstanding amount",
            "outstanding dues",
            "debt",
            "creditor",
            "debtor",
            "interest",
            "instalment",
            "installment",
            "default",
            "recovery",
            "recover",
            "payment",
            "money",
            "demand notice",
            "ಸಾಲ",
            "ಸಾಲ ಮರುಪಾವತಿ",
            "ಬಡ್ಡಿ",
            "ಬಾಕಿ ಹಣ",
            "ಬ್ಯಾಂಕ್ ಸಾಲ",
            "ಸಾಲ ವಸೂಲಾತಿ",
            "ಕಂತು",
            "ಸಾಲದ ನೋಟಿಸ್",
            "ಬಾಕಿ ಪಾವತಿ",
            "ऋण",
            "कर्ज",
            "उधार",
            "ब्याज",
            "बकाया राशि",
            "किस्त",
            "लोन रिकवरी",
            "ऋण वसूली नोटिस",
            "कर्ज वापसी",
        ],
        "strong_terms": [
            "loan",
            "loan agreement",
            "repayment",
            "borrower",
            "lender",
            "creditor",
            "debtor",
            "debt",
            "interest",
            "default",
            "money",
            "recovery",
            "payment",
            "ಸಾಲ",
            "ಸಾಲ ವಸೂಲಾತಿ",
            "ಬಾಕಿ ಹಣ",
            "ऋण वसूली",
            "कर्ज",
        ],
        "negative_terms": [
            "bailor",
            "bailee",
            "bailment",
            "pledge",
            "pawn",
            "redemption",
            "immovable property",
            "sale of immovable property",
            "easement",
        ],
        "queries": [
            "loan repayment recovery civil suit",
            "loan agreement repayment default recovery of money",
            "outstanding debt recovery creditor borrower legal notice",
            "ಸಾಲ ಮರುಪಾವತಿ ಮತ್ತು ಹಣ ವಸೂಲಾತಿ ಕಾನೂನು ನೋಟಿಸ್",
            "ऋण वसूली और कानूनी प्रक्रिया नोटिस",
        ],
    },

    "sarfaesi_banking": {
        "route": "commercial",
        "terms": [
            "sarfaesi",
            "sarfaesi act",
            "section 13(2)",
            "section 13(4)",
            "possession notice",
            "debt recovery tribunal",
            "drt",
            "secured creditor",
            "npa",
            "non performing asset",
            "rbi ombudsman",
            "banking regulation",
            "e-auction",
            "auction of mortgaged property",
            "security interest",
            "ಬ್ಯಾಂಕ್ ನೋಟಿಸ್",
            "ಸಾಲ ಜಪ್ತಿ",
            "ಡಿಆರ್‌ಟಿ",
            "ಆಸ್ತಿ ಜಪ್ತಿ",
            "ಸರ್ಫೇಸಿ",
            "ಬ್ಯಾಂಕ್ ವಸೂಲಾತಿ",
            "ಬ್ಯಾಂಕಿಂಗ್ ನಿಯಮಗಳು",
            "ನೋಟಿಸ್ 13(2)",
            "ಸರ್ಫೇಸಿ ಕಾಯ್ದೆ",
            "सरफेसी",
            "सरफेसी एक्ट",
            "बैंक नोटिस",
            "संपत्ति कुर्की",
            "डीआरटी",
            "एनपीए",
            "बैंक नीलामी",
            "कब्जा नोटिस",
        ],
        "strong_terms": [
            "sarfaesi",
            "section 13",
            "13(2)",
            "13(4)",
            "drt",
            "debt recovery tribunal",
            "npa",
            "ಸರ್ಫೇಸಿ",
            "ಡಿಆರ್‌ಟಿ",
            "सरफेसी",
            "डीआरटी",
        ],
        "queries": [
            "SARFAESI Act Section 13 2 13 4 possession notice borrower rights DRT appeal Section 17",
            "bank loan recovery SARFAESI auction notice 60 days representation",
            "Debt Recovery Tribunal appeal limitation 45 days against possession",
            "ಸರ್ಫೇಸಿ ಕಾಯ್ದೆ ಸೆಕ್ಷನ್ 13 ಬ್ಯಾಂಕ್ ನೋಟಿಸ್ ಮತ್ತು ಡಿಆರ್‌ಟಿ ಮೇಲ್ಮನವಿ",
            "सरफेसी एक्ट बैंक नोटिस और डीआरटी में अपील प्रक्रिया",
        ],
    },

    "contract_breach": {
        "route": "civil_contractual",
        "terms": [
            "contract",
            "agreement",
            "breach",
            "promise",
            "obligation",
            "consideration",
            "performance",
            "default",
            "damages",
            "compensation",
            "non compete",
            "specific relief",
            "ಕರಾರು ಉಲ್ಲಂಘನೆ",
            "ಒಪ್ಪಂದ ಮುರಿತ",
            "ನಷ್ಟ ಪರಿಹಾರ",
            "ಕರಾರು ಒಪ್ಪಂದ",
            "ವ್ಯಾಪಾರ ನಿರ್ಬಂಧ",
            "अनुबंध उल्लंघन",
            "करार",
            "मुआवजा",
            "हर्जाना",
            "समझौता तोड़ना",
        ],
        "queries": [
            "breach of contract damages Indian Contract Act Section 73 74",
            "contractual obligation non-compete clause Section 27 void",
            "agreement breach legal remedy specific performance",
        ],
    },

    "property_dispute": {
        "route": "property",
        "terms": [
            "property",
            "ownership",
            "possession",
            "title",
            "land",
            "building",
            "immovable",
            "partition",
            "boundary",
            "sale deed",
            "transfer of property",
            "mortgage",
            "lease",
            "tenant",
            "landlord",
            "eviction",
            "encroachment",
            "rent",
            "ಆಸ್ತಿ ವಿವಾದ",
            "ಜಮೀನು ತಕರಾರು",
            "ಖಾತೆ",
            "ನೋಂದಣಿ",
            "ಹಕ್ಕು",
            "ಪಾಲು",
            "ಕ್ರಯಪತ್ರ",
            "ಭೂ ಒತ್ತುವರಿ",
            "ಬಾಡಿಗೆದಾರ",
            "ತೆರವು",
            "ಮನೆ ಆಸ್ತಿ",
            "जमीन विवाद",
            "संपत्ति",
            "बैनामा",
            "कब्जा",
            "अवैध कब्जा",
            "बंटवारा",
            "किरायेदार",
            "मकान खाली",
            "भूमि विवाद",
        ],
        "queries": [
            "Transfer of Property Act sale mortgage lease gift Section 54 105",
            "property ownership possession illegal encroachment injunction",
            "landlord tenant eviction notice suit for possession",
            "ಆಸ್ತಿ ಮಾಲೀಕತ್ವ ಮತ್ತು ಒತ್ತುವರಿ ತಡೆ ಕಾನೂನು ಕ್ರಮ",
            "जमीन पर अवैध कब्जा और कानूनी उपचार",
        ],
    },

    "consumer_complaint": {
        "route": "consumer",
        "terms": [
            "consumer",
            "defective",
            "defect",
            "deficiency in service",
            "unfair trade practice",
            "district commission",
            "product liability",
            "warranty",
            "guarantee",
            "replacement",
            "refund",
            "consumer court",
            "consumer forum",
            "consumer protection",
            "e-daakhil",
            "ಗ್ರಾಹಕ ದೂರು",
            "ದೋಷಪೂರಿತ ವಸ್ತು",
            "ಗ್ರಾಹಕ ನ್ಯಾಯಾಲಯ",
            "ಪರಿಹಾರ",
            "ಕಳಪೆ ಸೇವೆ",
            "ವಾರಂಟಿ",
            "ಹಣ ಮರುಪಾವತಿ",
            "ಗ್ರಾಹಕ ರಕ್ಷಣೆ",
            "उपभोक्ता शिकायत",
            "खराब सामान",
            "घटिया सेवा",
            "उपभोक्ता अदालत",
            "उपभोक्ता फोरम",
            "वारंटी",
            "रिफंड",
            "ई-दाखिल",
        ],
        "queries": [
            "Consumer Protection Act 2019 defect goods deficiency service Section 35",
            "consumer complaint district commission refund replacement compensation",
            "unfair trade practice product liability consumer rights e-daakhil",
            "ಗ್ರಾಹಕ ಸಂರಕ್ಷಣಾ ಕಾಯ್ದೆ ದೋಷಪೂರಿತ ಸೇವೆ ಪರಿಹಾರ",
            "उपभोक्ता संरक्षण अधिनियम खराब माल और सेवा में कमी शिकायत",
        ],
    },

    "cheque_bounce": {
        "route": "commercial",
        "terms": [
            "cheque",
            "check",
            "dishonour",
            "dishonored",
            "dishonoured",
            "bounce",
            "bounced",
            "bouncing",
            "138",
            "section 138",
            "insufficient funds",
            "funds insufficient",
            "drawer",
            "payee",
            "notice period 15 days",
            "interim compensation",
            "statutory notice",
            "ಚೆಕ್",
            "ಚೆಕ್ ಬೌನ್ಸ್",
            "ಬೌನ್ಸ್",
            "ಅಮಾನ್ಯ",
            "ಖಾತೆಯಲ್ಲಿ ಹಣವಿಲ್ಲ",
            "ಚೆಕ್ ತಿರಸ್ಕಾರ",
            "ಸೆಕ್ಷನ್ 138",
            "15 ದಿನಗಳ ನೋಟಿಸ್",
            "चेक",
            "चेक बाउंस",
            "बाउंस",
            "खाते में अपर्याप्त राशि",
            "धारा 138",
            "15 दिन का कानूनी नोटिस",
            "चेक अनादरण",
        ],
        "strong_terms": [
            "138",
            "section 138",
            "dishonour of cheque",
            "cheque bounce",
            "funds insufficient",
            "insufficiency of funds",
            "ಚೆಕ್ ಬೌನ್ಸ್",
            "ಸೆಕ್ಷನ್ 138",
            "चेक बाउंस",
            "धारा 138",
        ],
        "queries": [
            "Section 138 Dishonour of cheque for insufficiency of funds Negotiable Instruments Act",
            "cheque bounce punishment imprisonment up to two years fine twice amount",
            "dishonour of cheque statutory notice 15 days payment demand 30 days filing",
            "ಚೆಕ್ ಬೌನ್ಸ್ ಪ್ರಕರಣ ಸೆಕ್ಷನ್ 138 ಶಿಕ್ಷೆ ಮತ್ತು ಕಾನೂನು ನೋಟಿಸ್",
            "चेक बाउंस धारा 138 नोटिस और 2 साल की सजा प्रावधान",
        ],
    },

    "corporate_company_law": {
        "route": "commercial",
        "terms": [
            "company",
            "companies act",
            "director",
            "directors",
            "duties of directors",
            "board of directors",
            "independent director",
            "corporate fraud",
            "section 447",
            "section 166",
            "section 135",
            "csr",
            "corporate social responsibility",
            "oppression",
            "mismanagement",
            "section 241",
            "section 242",
            "nclt",
            "national company law tribunal",
            "roc",
            "registrar of companies",
            "related party transactions",
            "section 188",
            "section 454",
            "dormant company",
            "ಕಂಪನಿ ಕಾಯ್ದೆ",
            "ನಿರ್ದೇಶಕರ ಕರ್ತವ್ಯ",
            "ಕಾರ್ಪೊರೇಟ್ ವಂಚನೆ",
            "ಸೆಕ್ಷನ್ 447",
            "ಸೆಕ್ಷನ್ 166",
            "ಎನ್‌ಸಿಎಲ್‌ಟಿ",
            "कंपनी अधिनियम",
            "निदेशक के कर्तव्य",
            "कॉर्पोरेट धोखाधड़ी",
            "धारा 447",
            "धारा 166",
            "एनसीएलटी",
        ],
        "strong_terms": [
            "companies act",
            "section 447",
            "section 166",
            "section 135",
            "section 241",
            "nclt",
            "duties of directors",
            "corporate fraud",
            "ಕಂಪನಿ ಕಾಯ್ದೆ",
            "कंपनी अधिनियम",
        ],
        "queries": [
            "Companies Act 2013 Section 447 punishment for fraud imprisonment fine",
            "Duties of directors Section 166 Companies Act 2013 fiduciary duty",
            "Section 135 Corporate Social Responsibility CSR spending net profit",
            "Section 241 242 Oppression and mismanagement NCLT application",
            "ಕಂಪನಿ ಕಾಯ್ದೆ 2013 ಸೆಕ್ಷನ್ 447 ವಂಚನೆ ಶಿಕ್ಷೆ ಮತ್ತು ನಿರ್ದೇಶಕರ ಕರ್ತವ್ಯ",
            "कंपनी अधिनियम 2013 धारा 447 कॉर्पोरेट फ्रॉड और सजा",
        ],
    },

    "cyber_crime": {
        "route": "cyber",
        "terms": [
            "cyber",
            "cyber crime",
            "online fraud",
            "upi fraud",
            "otp scam",
            "phishing",
            "hacked",
            "unauthorized transfer",
            "identity theft",
            "it act",
            "section 66",
            "section 66c",
            "section 66d",
            "cyber cell",
            "cybercrime.gov.in",
            "1930 helpline",
            "ಸೈಬರ್ ಅಪರಾಧ",
            "ಆನ್‌ಲೈನ್ ವಂಚನೆ",
            "ಯುಪಿಐ ಫ್ರಾಡ್",
            "ಓಟಿಪಿ ವಂಚನೆ",
            "ಹ್ಯಾಕಿಂಗ್",
            "ಅನಧಿಕೃತ ಹಣ ವರ್ಗಾವಣೆ",
            "ಸೈಬರ್ ಕ್ರೈಮ್ ಠಾಣೆ",
            "1930 ಸಹಾಯವಾಣಿ",
            "ಐಟಿ ಕಾಯ್ದೆ",
            "साइबर अपराध",
            "ऑनलाइन ठगी",
            "यूपीआई फ्रॉड",
            "ओटीपी फ्रॉड",
            "बैंक खाता हैक",
            "अनधिकृत ट्रांसफर",
            "साइबर सेल",
            "1930 हेल्पलाइन",
            "आईटी एक्ट",
        ],
        "strong_terms": [
            "cyber",
            "phishing",
            "upi fraud",
            "section 66",
            "it act",
            "ಸೈಬರ್",
            "ಯುಪಿಐ",
            "साइबर",
            "यूपीआई",
        ],
        "queries": [
            "Information Technology Act Section 66 66C 66D cheating by personation computer resource",
            "cyber financial fraud unauthorized UPI transfer 1930 helpline complaint",
            "cyber crime cell complaint online financial fraud refund freezing beneficiary account",
            "ಸೈಬರ್ ಅಪರಾಧ ಆನ್‌ಲೈನ್ ಬ್ಯಾಂಕಿಂಗ್ ವಂಚನೆ ಐಟಿ ಕಾಯ್ದೆ",
            "साइबर अपराध ऑनलाइन फ्रॉड आईटी एक्ट धारा 66डी और शिकायत",
        ],
    },

    "employment_dispute": {
        "route": "employment",
        "terms": [
            "employment",
            "salary",
            "unpaid salary",
            "gratuity",
            "provident fund",
            "pf",
            "wrongful termination",
            "resignation",
            "notice pay",
            "severance",
            "labour court",
            "industrial dispute",
            "payment of wages",
            "ವೇತನ",
            "ಸಂಬಳ ಬಾಕಿ",
            "ಅಕ್ರಮ ವಜಾ",
            "ಉಪಧನ",
            "ಗ್ರಾಚ್ಯುಟಿ",
            "ಪಿಎಫ್",
            "ನೌಕರಿ ವಿವಾದ",
            "ಕಾರ್ಮಿಕ ನ್ಯಾಯಾಲಯ",
            "ನೋಟಿಸ್ ವೇತನ",
            "वेतन",
            "सैलरी बकाया",
            "गलत तरीके से बर्खास्तगी",
            "ग्रेच्युटी",
            "पीएफ",
            "नौकरी विवाद",
            "श्रम न्यायालय",
            "नोटिस पे",
        ],
        "queries": [
            "Payment of Gratuity Act Code on Wages unpaid salary wrongful termination",
            "employment dispute labour commissioner complaint gratuity claim",
            "ಸಂಬಳ ಬಾಕಿ ಮತ್ತು ಗ್ರಾಚ್ಯುಟಿ ಕಾರ್ಮಿಕ ಇಲಾಖೆ ದೂರು",
            "वेतन बकाया और ग्रेच्युटी श्रम न्यायालय में दावा",
        ],
    },

    "domestic_violence": {
        "route": "family",
        "terms": [
            "domestic violence",
            "protection order",
            "residence order",
            "shared household",
            "monetary relief",
            "physical abuse",
            "emotional abuse",
            "verbal abuse",
            "economic abuse",
            "harassment",
            "in-laws",
            "protection officer",
            "aggrieved person",
            "stridhan",
            "ಗೃಹಹಿಂಸೆ",
            "ದೌರ್ಜನ್ಯ",
            "ಕೌಟುಂಬಿಕ ದೌರ್ಜನ್ಯ",
            "ವರದಕ್ಷಿಣೆ ಕಿರುಕುಳ",
            "ಘರೇಲು ಹಿಂಸಾ",
            "घरेलू हिंसा",
            "दहेज प्रताड़ना",
            "पत्नी के साथ मारपीट",
        ],
        "queries": [
            "protection of women from domestic violence act",
            "domestic violence protection order residence shared household",
            "domestic abuse monetary relief maintenance magistrate",
            "ಗೃಹಹಿಂಸೆ ಕಾಯ್ದೆ ರಕ್ಷಣಾ ಆದೇಶ ಮತ್ತು ಪರಿಹಾರ",
            "घरेलू हिंसा संरक्षण आदेश और भरण पोषण",
        ],
    },

    "marriage_divorce": {
        "route": "family",
        "terms": [
            "marriage",
            "divorce",
            "restitution of conjugal rights",
            "mutual consent",
            "alimony",
            "maintenance",
            "hindu marriage act",
            "judicial separation",
            "cruelty",
            "desertion",
            "child custody",
            "permanent alimony",
        ],
        "queries": [
            "hindu marriage act divorce mutual consent",
            "marriage dissolution cruelty desertion maintenance alimony",
            "restitution of conjugal rights child custody",
        ],
    },

    "cyber_crime": {
        "route": "cyber",
        "terms": [
            "cyber",
            "cybercrime",
            "identity theft",
            "online fraud",
            "hacking",
            "hacked",
            "unauthorized access",
            "computer virus",
            "phishing",
            "privacy violation",
            "information technology act",
            "section 66",
            "section 66c",
            "section 66d",
            "electronic data",
            "data theft",
            "online scam",
        ],
        "queries": [
            "information technology act cyber crime penalty",
            "identity theft cheating personation computer resource",
            "unauthorized access computer system online fraud section 66",
        ],
    },

    "employment_dispute": {
        "route": "employment",
        "terms": [
            "employee",
            "employer",
            "employment",
            "salary",
            "wages",
            "termination",
            "dismissal",
            "workplace",
            "labour",
            "labor",
            "gratuity",
            "payment of gratuity",
            "continuous service",
            "superannuation",
            "resignation gratuity",
            "unpaid salary",
        ],
        "queries": [
            "payment of gratuity act continuous service 5 years",
            "employment dispute wages termination gratuity recovery",
            "employee employer labour law salary delay",
        ],
    },

    "legal_aid": {
        "route": "legal_aid",
        "terms": [
            "legal aid",
            "free legal aid",
            "free lawyer",
            "nalsa",
            "slsa",
            "dlsa",
            "lok adalat",
            "legal services",
            "poor citizen",
            "free legal assistance",
            "legal services authority",
            "taluk legal services",
            "permanent lok adalat",
        ],
        "queries": [
            "legal services authorities act free legal aid eligibility",
            "lok adalat award compromise settlement nalsa dlsa",
            "criteria for giving free legal services section 12",
        ],
    },

    "traffic_motor_vehicle": {
        "route": "traffic",
        "terms": [
            "traffic",
            "traffic light",
            "red light",
            "jump red light",
            "jumping traffic light",
            "jump the traffic light",
            "jump traffic signal",
            "traffic signal",
            "signal jump",
            "traffic violation",
            "motor vehicle",
            "driving",
            "driving license",
            "driving without license",
            "drunk driving",
            "alcohol driving",
            "overspeeding",
            "speed limit",
            "helmet",
            "without helmet",
            "seatbelt",
            "challan",
            "traffic challan",
            "traffic police",
            "motor vehicles act",
            "section 119",
            "section 184",
            "section 185",
            "section 181",
            "section 177",
            "section 194d",
            "juvenile driving",
            "ಸಂಚಾರ ನಿಯಮ",
            "ಸಿಗ್ನಲ್",
            "ಕೆಂಪು ದೀಪ",
            "ಟ್ರಾಫಿಕ್ ಸಿಗ್ನಲ್",
            "ಹೆಲ್ಮೆಟ್",
            "ಸಂಚಾರ ದಂಡ",
            "ಚಾಲನಾ ಪರವಾನಗಿ",
            "ಮದ್ಯಪಾನ ಚಾಲನೆ",
            "यातायात",
            "ट्रैफिक लाइट",
            "रेड लाइट जंप",
            "सिग्नल जंप",
            "ट्रैफिक चालान",
            "बिना हेलमेट",
            "शराब पीकर गाड़ी चलाना",
            "बिना लाइसेंस",
        ],
        "strong_terms": [
            "traffic light",
            "red light",
            "traffic signal",
            "jump the traffic light",
            "jump red light",
            "jumping traffic light",
            "signal jump",
            "motor vehicles act",
            "section 184",
            "drunk driving",
            "driving without license",
            "ट्रैफिक लाइट",
            "ಸಿಗ್ನಲ್",
        ],
        "queries": [
            "Motor Vehicles Act Section 184 driving dangerously jumping red light traffic signal penalty fine",
            "Section 119 duty to obey traffic signs red light signals Motor Vehicles Act",
            "penalty for jumping traffic light fine driving dangerously section 184",
            "driving without valid license drunk driving overspeeding penalty motor vehicles act",
        ],
    },

    "rti": {
        "route": "governance_rti",
        "terms": [
            "rti",
            "right to information",
            "public information officer",
            "cpio",
            "spio",
            "information commission",
            "first appeal",
            "second appeal",
            "30 days",
            "48 hours life liberty",
            "information request",
            "section 6",
            "section 7",
            "section 8",
            "section 19",
            "section 20",
            "ಮಾಹಿತಿ ಹಕ್ಕು",
            "ಆರ್.ಟಿ.ಐ",
            "ಅರ್ಜಿ",
            "सूचना का अधिकार",
            "आरटीआई",
            "प्रथम अपील",
        ],
        "queries": [
            "Right to Information Act 2005 Section 6 request information CPIO 30 days",
            "RTI first appeal second appeal section 19 penalty on PIO section 20",
            "exemption from disclosure section 8 Right to Information Act",
        ],
    },

    "pocso": {
        "route": "children_pocso",
        "terms": [
            "pocso",
            "child abuse",
            "sexual offence child",
            "penetrative sexual assault",
            "mandatory reporting",
            "child sexual assault",
            "section 3",
            "section 4",
            "section 19",
            "section 21",
            "ಪೋಕ್ಸೋ",
            "ಮಕ್ಕಳ ಮೇಲಿನ ಲೈಂಗಿಕ ದೌರ್ಜನ್ಯ",
            "पॉक्सो",
            "बाल यौन शोषण",
        ],
        "queries": [
            "POCSO Act 2012 penetrative sexual assault punishment section 4",
            "mandatory reporting of offences against children section 19 section 21",
        ],
    },

    "rera_real_estate": {
        "route": "real_estate",
        "terms": [
            "rera",
            "real estate",
            "builder delay",
            "flat possession",
            "possession delay",
            "carpet area",
            "defect liability",
            "5 years defect",
            "refund with interest",
            "builder compensation",
            "section 14",
            "section 18",
            "section 31",
            "ರೇರಾ",
            "ಫ್ಲಾಟ್ ಸ್ವಾಧೀನ ವಿಳಂಬ",
            "ರೆರಾ",
            "बिल्डर देरी",
            "कब्जा देरी",
            "रेरा",
        ],
        "queries": [
            "RERA Act 2016 Section 18 refund with interest compensation builder delay possession",
            "structural defect liability 5 years section 14 real estate regulation act",
        ],
    },

    "senior_citizens": {
        "route": "senior_citizens",
        "terms": [
            "senior citizen",
            "parents maintenance",
            "elderly",
            "maintenance of parents",
            "abandonment of senior citizen",
            "cancel gift deed",
            "property transfer void",
            "maintenance tribunal",
            "section 4",
            "section 9",
            "section 23",
            "section 24",
            "ಹಿರಿಯ ನಾಗರಿಕರು",
            "ಪೋಷಕರ ಪೋಷಣೆ",
            "ಆಸ್ತಿ ವರ್ಗಾವಣೆ ರದ್ದು",
            "वरिष्ठ नागरिक",
            "माता पिता भरण पोषण",
            "संपत्ति हस्तांतरण निरस्त",
        ],
        "queries": [
            "Maintenance and Welfare of Parents and Senior Citizens Act 2007 section 4 maintenance",
            "transfer of property void failure to maintain senior citizen section 23",
            "abandonment of parents penalty section 24 senior citizens act",
        ],
    },
}


# ============================================================
# LANGUAGE TERMS
# ============================================================

MULTILINGUAL_LEGAL_TERMS = {
    # Traffic
    "traffic": "traffic_motor_vehicle",
    "traffic light": "traffic_motor_vehicle",
    "red light": "traffic_motor_vehicle",
    "jump traffic light": "traffic_motor_vehicle",
    "jump the traffic light": "traffic_motor_vehicle",
    "jumping traffic light": "traffic_motor_vehicle",
    "red light jump": "traffic_motor_vehicle",
    "signal jump": "traffic_motor_vehicle",
    "drunk driving": "traffic_motor_vehicle",
    "driving without license": "traffic_motor_vehicle",
    "overspeeding": "traffic_motor_vehicle",
    "helmet": "traffic_motor_vehicle",
    "challan": "traffic_motor_vehicle",
    "ಸಂಚಾರ ನಿಯಮ": "traffic_motor_vehicle",
    "ಸಿಗ್ನಲ್": "traffic_motor_vehicle",
    "ಕೆಂಪು ದೀಪ": "traffic_motor_vehicle",
    "ಟ್ರಾಫಿಕ್": "traffic_motor_vehicle",
    "ಹೆಲ್ಮೆಟ್": "traffic_motor_vehicle",
    "यातायात": "traffic_motor_vehicle",
    "ट्रैफिक लाइट": "traffic_motor_vehicle",
    "सिग्नल जंप": "traffic_motor_vehicle",
    "चालान": "traffic_motor_vehicle",

    # RTI
    "rti": "rti",
    "right to information": "rti",
    "ಮಾಹಿತಿ ಹಕ್ಕು": "rti",
    "सूचना का अधिकार": "rti",
    "आरटीआई": "rti",

    # POCSO
    "pocso": "pocso",
    "child abuse": "pocso",
    "ಪೋಕ್ಸೋ": "pocso",
    "पॉक्सो": "pocso",

    # RERA
    "rera": "rera_real_estate",
    "builder delay": "rera_real_estate",
    "possession delay": "rera_real_estate",
    "ರೇರಾ": "rera_real_estate",
    "रेरा": "rera_real_estate",

    # Senior Citizens
    "senior citizen": "senior_citizens",
    "parents maintenance": "senior_citizens",
    "ಹಿರಿಯ ನಾಗರಿಕರು": "senior_citizens",
    "वरिष्ठ नागरिक": "senior_citizens",

    # Hindi
    "चोरी": "theft",
    "कानूनी चोरी": "theft",
    "चोरी की सजा": "theft",
    "हत्या": "murder",
    "हत्या की सजा": "murder",
    "धोखाधड़ी": "cheating",
    "धोखा": "cheating",
    "चेक बाउंस": "cheque_bounce",
    "चेक अनादर": "cheque_bounce",
    "उपभोक्ता": "consumer_complaint",
    "घरेलू हिंसा": "domestic_violence",
    "तलाक": "marriage_divorce",
    "विवाह": "marriage_divorce",
    "साइबर": "cyber_crime",
    "साइबर अपराध": "cyber_crime",
    "ग्रेच्युटी": "employment_dispute",
    "मुफ्त कानूनी सहायता": "legal_aid",
    "लोक अदालत": "legal_aid",

    # Kannada
    "ಕಳ್ಳತನ": "theft",
    "ಕಳ್ಳತನಕ್ಕೆ ಶಿಕ್ಷೆ": "theft",
    "ಕೊಲೆ": "murder",
    "ಕೊಲೆಗೆ ಶಿಕ್ಷೆ": "murder",
    "ಮೋಸ": "cheating",
    "ವಂಚನೆ": "cheating",
    "ಚೆಕ್ ಬೌನ್ಸ್": "cheque_bounce",
    "ಗ್ರಾಹಕ": "consumer_complaint",
    "ಗೃಹ ಹಿಂಸಾಚಾರ": "domestic_violence",
    "ವಿಚ್ಛೇದನ": "marriage_divorce",
    "ಮದುವೆ": "marriage_divorce",
    "ಸೈಬರ್ ಅಪರಾಧ": "cyber_crime",
    "ಗ್ರಾಚ್ಯುಟಿ": "employment_dispute",
    "ಉಚಿತ ಕಾನೂನು ನೆರವು": "legal_aid",
    "ಲೋಕ ಅದಾಲತ್": "legal_aid",
}


# ============================================================
# INTENT PHRASES
# ============================================================

PUNISHMENT_PHRASES = [
    "punishment",
    "punishable",
    "punishment for",
    "penalty",
    "sentence",
    "fine",
    "how many years",
    "how long imprisonment",

    "सजा",
    "दंड",
    "सज़ा",

    "ಶಿಕ್ಷೆ",
    "ದಂಡ",
]

DEFINITION_PHRASES = [
    "what is",
    "what does",
    "define",
    "definition",
    "meaning of",
    "means",

    "क्या है",
    "क्या होता है",
    "परिभाषा",

    "ಎಂದರೇನು",
    "ಅರ್ಥವೇನು",
    "ವ್ಯಾಖ್ಯಾನ",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def _normalize_text(
    text: Any,
) -> str:

    if text is None:
        return ""

    value = str(
        text
    ).strip().lower()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def _tokenize(
    text: Any,
) -> List[str]:

    if not text:
        return []

    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(
            str(text)
        )
    ]


def _json_safe(
    value: Any,
) -> Any:

    if isinstance(
        value,
        set,
    ):

        return sorted(
            str(item)
            for item in value
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
                _json_safe(
                    item
                )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            _json_safe(
                item
            )
            for item in value
        ]

    return value


# ============================================================
# DOCUMENT CACHE
# ============================================================

def _get_documents() -> Tuple[
    List[str],
    List[dict],
]:

    global _DOCUMENT_CACHE

    if _DOCUMENT_CACHE is None:

        documents, metadatas = (
            get_all_documents()
        )

        _DOCUMENT_CACHE = (
            list(documents),
            list(metadatas),
        )

    return _DOCUMENT_CACHE


def clear_retriever_cache() -> None:

    global _DOCUMENT_CACHE
    global _BM25_CACHE

    _DOCUMENT_CACHE = None
    _BM25_CACHE = None


# ============================================================
# BM25 IMPLEMENTATION
# ============================================================

def _build_bm25_index() -> Dict[str, Any]:

    global _BM25_CACHE

    documents, _ = _get_documents()

    if (
        _BM25_CACHE is not None
        and _BM25_CACHE.get(
            "document_count"
        ) == len(documents)
    ):

        return _BM25_CACHE

    tokenized_docs = [
        _tokenize(
            document
        )
        for document in documents
    ]

    doc_lengths = [
        len(tokens)
        for tokens in tokenized_docs
    ]

    average_length = (
        sum(doc_lengths)
        / len(doc_lengths)
        if doc_lengths
        else 1.0
    )

    document_frequency = Counter()

    for tokens in tokenized_docs:

        unique_tokens = set(
            tokens
        )

        for token in unique_tokens:

            document_frequency[
                token
            ] += 1

    _BM25_CACHE = {
        "tokens":
            tokenized_docs,

        "doc_lengths":
            doc_lengths,

        "average_length":
            average_length,

        "document_frequency":
            document_frequency,

        "document_count":
            len(documents),
    }

    return _BM25_CACHE


def _bm25_score_document(
    query_tokens: List[str],
    doc_index: int,
    bm25_index: Dict[str, Any],
) -> float:

    if not query_tokens:
        return 0.0

    tokens = bm25_index[
        "tokens"
    ][doc_index]

    if not tokens:
        return 0.0

    counts = Counter(
        tokens
    )

    document_count = (
        bm25_index[
            "document_count"
        ]
    )

    document_frequency = (
        bm25_index[
            "document_frequency"
        ]
    )

    doc_length = (
        bm25_index[
            "doc_lengths"
        ][doc_index]
    )

    average_length = max(
        float(
            bm25_index[
                "average_length"
            ]
        ),
        1.0,
    )

    k1 = 1.5
    b = 0.75

    score = 0.0

    for term in query_tokens:

        tf = counts.get(
            term,
            0,
        )

        if tf <= 0:
            continue

        df = document_frequency.get(
            term,
            0,
        )

        if df <= 0:
            continue

        idf = math.log(
            1.0
            +
            (
                document_count
                - df
                + 0.5
            )
            /
            (
                df
                + 0.5
            )
        )

        denominator = (
            tf
            +
            k1
            *
            (
                1.0
                -
                b
                +
                b
                *
                (
                    doc_length
                    /
                    average_length
                )
            )
        )

        score += (
            idf
            *
            (
                tf
                *
                (k1 + 1.0)
                /
                max(
                    denominator,
                    1e-9,
                )
            )
        )

    return float(
        score
    )


def _min_max_normalize(
    values: List[float],
) -> List[float]:

    if not values:
        return []

    low = min(
        values
    )

    high = max(
        values
    )

    if (
        math.isclose(
            low,
            high,
        )
    ):

        return [
            0.0
            for _ in values
        ]

    return [
        (
            value
            - low
        )
        /
        (
            high
            - low
        )
        for value in values
    ]


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_query_intent(
    query: str,
) -> str:

    original = str(
        query or ""
    ).strip()

    normalized = _normalize_text(
        original
    )

    # Punishment first because phrases like
    # "what is the punishment" also contain "what is".
    for phrase in PUNISHMENT_PHRASES:

        if phrase in normalized:

            return "punishment"

    for phrase in DEFINITION_PHRASES:

        if phrase in normalized:

            return "definition"

    return "general"


# Compatibility alias used elsewhere in JanNyaya.
def detect_fact_intent(
    query: str,
) -> str:

    return detect_query_intent(
        query
    )


# ============================================================
# LEGAL TERM EXTRACTION
# ============================================================

def extract_query_legal_term(
    query: str,
) -> str:

    if not query:
        return ""

    original = str(
        query
    ).strip()

    if not original:
        return ""

    normalized = _normalize_text(
        original
    )

    # 1. Multilingual direct matches (sorted by descending length)
    for phrase in sorted(
        MULTILINGUAL_LEGAL_TERMS.keys(),
        key=len,
        reverse=True,
    ):
        if phrase in original:
            return MULTILINGUAL_LEGAL_TERMS[phrase]

    # 2. Collect all terms across all topics sorted by descending length.
    # This guarantees 'identity theft' matches 'cyber_crime' before 'theft' matches 'theft',
    # and 'immovable property' / 'sale deed' matches 'property_dispute' before generic words.
    all_topic_terms: List[Tuple[str, str]] = []
    for topic, cfg in TOPIC_CONFIG.items():
        for term in cfg.get("terms", []):
            norm_term = _normalize_text(term)
            if norm_term:
                all_topic_terms.append((norm_term, topic))

    # Sort descending by length of term
    all_topic_terms.sort(key=lambda item: len(item[0]), reverse=True)

    for term, topic in all_topic_terms:
        # Check word boundary pattern
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, normalized, re.IGNORECASE):
            return topic

    return ""


# Compatibility alias used elsewhere in JanNyaya.
def detect_legal_term(
    query: str,
) -> str:

    return extract_query_legal_term(
        query
    )


# ============================================================
# ROUTE DETECTION
# ============================================================

def detect_query_route(
    query: str,
) -> str:

    legal_term = (
        extract_query_legal_term(
            query
        )
    )

    configuration = TOPIC_CONFIG.get(
        legal_term,
        {},
    )

    return str(
        configuration.get(
            "route",
            "general",
        )
    )


# ============================================================
# METADATA HELPERS
# ============================================================

def _metadata(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    value = result.get(
        "metadata",
        {},
    )

    if isinstance(
        value,
        dict,
    ):

        return value

    return {}


def _get_section(
    metadata: Dict[str, Any],
) -> str:
    sec = str(
        metadata.get(
            "section_number",
            metadata.get(
                "section",
                "",
            ),
        )
        or ""
    ).strip()
    if sec and sec.lower() not in ("none", "unknown", ""):
        return sec

    title = str(metadata.get("section_title") or metadata.get("title") or "")
    m = re.match(r"^(?:Section\s*)?(\d+[A-Za-z]?)\b", title, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _get_section_title(
    metadata: Dict[str, Any],
) -> str:

    return str(
        metadata.get(
            "section_title",
            "",
        )
        or ""
    ).strip()


def _get_route(
    metadata: Dict[str, Any],
) -> str:

    return _normalize_text(
        metadata.get(
            "route",
            "",
        )
    )


def _result_key(
    result: Dict[str, Any],
) -> str:

    metadata = _metadata(
        result
    )

    document_id = str(
        metadata.get(
            "document_id",
            "",
        )
    )

    chunk_index = str(
        metadata.get(
            "chunk_index",
            "",
        )
    )

    source = str(
        metadata.get(
            "source",
            "",
        )
    )

    section = _get_section(
        metadata
    )

    document = str(
        result.get(
            "document",
            "",
        )
    )

    return "|".join(
        [
            document_id,
            chunk_index,
            source,
            section,
            document[:100],
        ]
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    top_k: int = DEFAULT_SEMANTIC_K,
) -> List[Dict[str, Any]]:

    if not query or not query.strip():
        return []

    documents, metadatas = (
        _get_documents()
    )

    if not documents:
        return []

    embedding = create_embedding(
        query
    )

    if not embedding:
        return []

    raw = search_by_embedding(
        embedding,
        n_results=min(
            int(top_k),
            len(documents),
        ),
    )

    ids = raw.get(
        "ids",
        [[]],
    )

    documents_result = raw.get(
        "documents",
        [[]],
    )

    metadatas_result = raw.get(
        "metadatas",
        [[]],
    )

    distances = raw.get(
        "distances",
        [[]],
    )

    ids = (
        ids[0]
        if ids
        else []
    )

    documents_result = (
        documents_result[0]
        if documents_result
        else []
    )

    metadatas_result = (
        metadatas_result[0]
        if metadatas_result
        else []
    )

    distances = (
        distances[0]
        if distances
        else []
    )

    results = []

    for index, document in enumerate(
        documents_result
    ):

        metadata = (
            metadatas_result[index]
            if index
            < len(
                metadatas_result
            )
            else {}
        )

        distance = (
            float(
                distances[index]
            )
            if index
            < len(
                distances
            )
            else 1.0
        )

        # Chroma distance is lower when closer.
        # Convert it into a simple descending score.
        similarity_score = (
            1.0
            /
            (
                1.0
                +
                max(
                    distance,
                    0.0,
                )
            )
        )

        results.append(
            {
                "id":
                    ids[index]
                    if index
                    < len(ids)
                    else "",

                "document":
                    document,

                "metadata":
                    metadata
                    if isinstance(
                        metadata,
                        dict,
                    )
                    else {},

                "distance":
                    distance,

                "score":
                    similarity_score,

                "method":
                    "semantic",
            }
        )

    return results


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query: str,
    documents: List[str],
    metadatas: List[dict],
    top_k: int = DEFAULT_BM25_K,
) -> List[Dict[str, Any]]:

    if not query:
        return []

    if not documents:
        return []

    bm25_index = (
        _build_bm25_index()
    )

    query_tokens = _tokenize(
        query
    )

    scored = []

    for index in range(
        len(documents)
    ):

        score = (
            _bm25_score_document(
                query_tokens,
                index,
                bm25_index,
            )
        )

        if score <= 0.0:
            continue

        scored.append(
            (
                index,
                score,
            )
        )

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    results = []

    for index, score in scored[
        :int(top_k)
    ]:

        results.append(
            {
                "document":
                    documents[index],

                "metadata":
                    metadatas[index],

                "score":
                    float(score),

                "method":
                    "bm25",
            }
        )

    return results


# ============================================================
# TOPIC SCORE
# ============================================================

def _topic_score(
    query_term: str,
    candidate: Dict[str, Any],
) -> float:

    if not query_term:
        return 0.0

    configuration = (
        TOPIC_CONFIG.get(
            query_term,
            {},
        )
    )

    if not configuration:
        return 0.0

    metadata = _metadata(
        candidate
    )

    title = _normalize_text(
        _get_section_title(
            metadata
        )
    )

    document = _normalize_text(
        candidate.get(
            "document",
            "",
        )
    )

    source_title = _normalize_text(
        " ".join(
            [
                str(
                    metadata.get(
                        "title",
                        "",
                    )
                ),
                str(
                    metadata.get(
                        "act_name",
                        "",
                    )
                ),
            ]
        )
    )

    haystack = " ".join(
        [
            title,
            document,
            source_title,
        ]
    )

    score = 0.0

    # Route.
    expected_route = str(
        configuration.get(
            "route",
            "general",
        )
    )

    actual_route = _get_route(
        metadata
    )

    if (
        expected_route
        and actual_route
        == expected_route
    ):

        score += 7.0

    elif (
        expected_route
        and actual_route
        and actual_route
        != expected_route
    ):

        score -= PENALTY_WRONG_ROUTE

    # Positive terms.
    for term in configuration.get(
        "terms",
        [],
    ):

        normalized_term = (
            _normalize_text(
                term
            )
        )

        if not normalized_term:
            continue

        if normalized_term in haystack:

            score += 1.5

    # Strong terms.
    for term in configuration.get(
        "strong_terms",
        [],
    ):

        normalized_term = (
            _normalize_text(
                term
            )
        )

        if (
            normalized_term
            and normalized_term
            in haystack
        ):

            score += 3.0

    # Negative terms.
    for term in configuration.get(
        "negative_terms",
        [],
    ):

        normalized_term = (
            _normalize_text(
                term
            )
        )

        if (
            normalized_term
            and normalized_term
            in haystack
        ):

            score -= 4.0

    # Title match is more valuable than body-only match.
    for term in configuration.get(
        "terms",
        [],
    ):

        normalized_term = (
            _normalize_text(
                term
            )
        )

    # Penalize amendment/footnote fragments heavily
    lower_haystack = haystack.lower()
    if any(
        amendment_term in lower_haystack
        for amendment_term in [
            "subs. by the a.o.",
            "subs. by act",
            "ins. by act",
            "repealed by",
            "a.o. 1950",
            "act 24 of 1917",
            "footnote",
        ]
    ):
        score -= 25.0

    # Specific false-positive penalties for loan recovery
    if query_term == "loan_recovery":
        sec_num = _get_section(metadata)
        if sec_num == "126" and not any(k in lower_haystack for k in ["guarantor", "surety", "contract of guarantee"]):
            score -= 20.0
        if sec_num == "60" and not any(k in lower_haystack for k in ["salary attachment", "execution decree", "warrant"]):
            score -= 20.0

    return score



# ============================================================
# LEGAL FEATURE SCORING
# ============================================================

def _legal_feature_score(
    query: str,
    query_term: str,
    query_intent: str,
    candidate: Dict[str, Any],
) -> float:

    metadata = _metadata(
        candidate
    )

    document = _normalize_text(
        candidate.get(
            "document",
            "",
        )
    )

    title = _normalize_text(
        _get_section_title(
            metadata
        )
    )

    query_normalized = _normalize_text(
        query
    )

    score = 0.0

    # Direct query token overlap.
    query_tokens = set(
        _tokenize(
            query_normalized
        )
    )

    title_tokens = set(
        _tokenize(
            title
        )
    )

    document_tokens = set(
        _tokenize(
            document
        )
    )

    if query_tokens:

        overlap_title = (
            len(
                query_tokens
                &
                title_tokens
            )
            /
            max(
                len(
                    query_tokens
                ),
                1,
            )
        )

        overlap_document = (
            len(
                query_tokens
                &
                document_tokens
            )
            /
            max(
                len(
                    query_tokens
                ),
                1,
            )
        )

        score += (
            overlap_title
            * 5.0
        )

        score += (
            overlap_document
            * 2.0
        )

    # Legal topic score.
    score += (
        _topic_score(
            query_term,
            candidate,
        )
        * WEIGHT_TOPIC
    )

    # Punishment intent.
    if query_intent == "punishment":

        if any(
            phrase in document
            for phrase in (
                "shall be punished",
                "punished with",
                "imprisonment",
                "liable to fine",
                "fine",
            )
        ):

            score += (
                BONUS_PUNISHMENT_PHRASE
            )

    # Definition intent.
    if query_intent == "definition":

        if any(
            phrase in document
            for phrase in (
                "is said to",
                "means",
                "defined as",
                "whoever",
            )
        ):

            score += (
                BONUS_DEFINITION_PHRASE
            )

    # Query-term title match.
    if query_term:

        normalized_title = (
            _normalize_text(
                title
            )
        )

        if (
            query_term
            in normalized_title
        ):

            score += (
                BONUS_DIRECT_TITLE
            )

    return float(
        score
    )


# ============================================================
# SECTION EXPANSION
# ============================================================

def _find_section_documents(
    section: str,
    documents: List[str],
    metadatas: List[dict],
) -> List[Dict[str, Any]]:

    if not section:
        return []

    results = []

    for index, metadata in enumerate(
        metadatas
    ):

        if not isinstance(
            metadata,
            dict,
        ):
            continue

        if (
            str(
                metadata.get(
                    "section_number",
                    metadata.get(
                        "section",
                        "",
                    ),
                )
            ).strip()
            != str(
                section
            ).strip()
        ):
            continue

        results.append(
            {
                "document":
                    documents[index],

                "metadata":
                    metadata,

                "method":
                    "section_expansion",

                "score":
                    0.0,

                "section_expansion":
                    True,
            }
        )

    return results


# ============================================================
# REMOVE DUPLICATE SECTIONS
# ============================================================

def _deduplicate_sections(
    results: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:

    output = []

    seen = set()

    for result in results:

        metadata = _metadata(
            result
        )

        act = str(
            metadata.get(
                "title",
                metadata.get(
                    "act_name",
                    "",
                ),
            )
        ).strip()

        source = str(
            metadata.get(
                "source",
                "",
            )
        ).strip()

        section = _get_section(
            metadata
        )

        # If section is available, one record per Act/source/section.
        if section:

            key = (
                act,
                source,
                section,
            )

        else:

            key = (
                act,
                source,
                str(
                    result.get(
                        "document",
                        "",
                    )
                )[:120],
            )

        if key in seen:
            continue

        seen.add(
            key
        )

        output.append(
            result
        )

    return output


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    semantic_k: int = DEFAULT_SEMANTIC_K,
    bm25_k: int = DEFAULT_BM25_K,
    final_k: Optional[int] = None,
) -> List[
    Dict[str, Any]
]:

    if not query or not query.strip():
        return []

    query = query.strip()

    if final_k is None:
        final_k = DEFAULT_FINAL_K

    final_k = max(
        int(final_k),
        1,
    )

    print()
    print(
        "Running legal retrieval..."
    )

    query_intent = (
        detect_query_intent(
            query
        )
    )

    query_term = (
        extract_query_legal_term(
            query
        )
    )

    query_route = (
        detect_query_route(
            query
        )
    )

    print(
        f"Detected intent: {query_intent}"
    )

    print(
        f"Detected legal term: {query_term}"
    )

    print(
        f"Detected route: {query_route}"
    )

    documents, metadatas = (
        _get_documents()
    )

    if not documents:
        return []

    # ========================================================
    # SEMANTIC
    # ========================================================

    print(
        "Running semantic search..."
    )

    semantic_results = semantic_search(
        query,
        top_k=max(
            int(
                semantic_k
            ),
            DEFAULT_SEMANTIC_K,
        ),
    )

    # ========================================================
    # BM25
    # ========================================================

    print(
        "Running BM25 search..."
    )

    bm25_queries = [query]
    if query_term and query_term in TOPIC_CONFIG:
        bm25_queries.extend(TOPIC_CONFIG[query_term].get("queries", []))

    bm25_results = []
    seen_bm25_docs = set()
    for bq in bm25_queries:
        for res in bm25_search(
            bq,
            documents,
            metadatas,
            top_k=max(
                int(bm25_k),
                DEFAULT_BM25_K,
            ),
        ):
            doc_snippet = str(res.get("document", ""))[:120]
            if doc_snippet not in seen_bm25_docs:
                seen_bm25_docs.add(doc_snippet)
                bm25_results.append(res)

    # ========================================================
    # RRF FUSION
    # ========================================================

    fusion_scores: Dict[
        str,
        float,
    ] = {}

    result_data: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # Semantic results.
    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):

        key = _result_key(
            result
        )

        rrf_score = (
            1.0
            /
            (
                RRF_K
                + rank
            )
        )

        fusion_scores[key] = (
            fusion_scores.get(
                key,
                0.0,
            )
            + rrf_score
        )

        entry = result_data.setdefault(
            key,
            {
                "document":
                    result.get(
                        "document",
                        "",
                    ),

                "metadata":
                    result.get(
                        "metadata",
                        {},
                    ),

                "methods":
                    set(),
            },
        )

        entry[
            "methods"
        ].add(
            "semantic"
        )

        entry[
            "semantic_score"
        ] = float(
            result.get(
                "score",
                0.0,
            )
        )

        entry[
            "semantic_rank"
        ] = rank

    # BM25 results.
    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        key = _result_key(
            result
        )

        rrf_score = (
            1.0
            /
            (
                RRF_K
                + rank
            )
        )

        fusion_scores[key] = (
            fusion_scores.get(
                key,
                0.0,
            )
            + rrf_score
        )

        entry = result_data.setdefault(
            key,
            {
                "document":
                    result.get(
                        "document",
                        "",
                    ),

                "metadata":
                    result.get(
                        "metadata",
                        {},
                    ),

                "methods":
                    set(),
            },
        )

        entry[
            "methods"
        ].add(
            "bm25"
        )

        entry[
            "bm25_score"
        ] = float(
            result.get(
                "score",
                0.0,
            )
        )

        entry[
            "bm25_rank"
        ] = rank

    candidates = []

    for key, fusion_score in (
        fusion_scores.items()
    ):

        candidate = dict(
            result_data[
                key
            ]
        )

        candidate[
            "rrf_score"
        ] = float(
            fusion_score
        )

        candidates.append(
            candidate
        )

    if not candidates:
        return []

    # ========================================================
    # NORMALIZE SCORES
    # ========================================================

    bm25_values = [
        float(
            candidate.get(
                "bm25_score",
                0.0,
            )
        )
        for candidate in candidates
        if "bm25_score"
        in candidate
    ]

    semantic_values = [
        float(
            candidate.get(
                "semantic_score",
                0.0,
            )
        )
        for candidate in candidates
        if "semantic_score"
        in candidate
    ]

    normalized_bm25 = (
        _min_max_normalize(
            bm25_values
        )
    )

    normalized_semantic = (
        _min_max_normalize(
            semantic_values
        )
    )

    bm25_index = 0
    semantic_index = 0

    # ========================================================
    # LEGAL RERANKING
    # ========================================================

    for candidate in candidates:

        if "bm25_score" in candidate:

            candidate[
                "bm25_normalized"
            ] = (
                normalized_bm25[
                    bm25_index
                ]
            )

            bm25_index += 1

        else:

            candidate[
                "bm25_normalized"
            ] = 0.0

        if "semantic_score" in candidate:

            candidate[
                "semantic_normalized"
            ] = (
                normalized_semantic[
                    semantic_index
                ]
            )

            semantic_index += 1

        else:

            candidate[
                "semantic_normalized"
            ] = 0.0

        rrf_score = float(
            candidate.get(
                "rrf_score",
                0.0,
            )
        )

        bm25_score = float(
            candidate.get(
                "bm25_normalized",
                0.0,
            )
        )

        semantic_score = float(
            candidate.get(
                "semantic_normalized",
                0.0,
            )
        )

        legal_score = (
            rrf_score
            * WEIGHT_RRF
        )

        legal_score += (
            bm25_score
            * WEIGHT_BM25
        )

        legal_score += (
            semantic_score
            * WEIGHT_SEMANTIC
        )

        # ----------------------------------------------------
        # Topic score
        # ----------------------------------------------------

        topic_score = (
            _topic_score(
                query_term,
                candidate,
            )
        )

        legal_score += (
            topic_score
            * WEIGHT_TOPIC
        )

        # ----------------------------------------------------
        # Feature score
        # ----------------------------------------------------

        feature_score = (
            _legal_feature_score(
                query,
                query_term,
                query_intent,
                candidate,
            )
        )

        legal_score += (
            feature_score
            * 0.50
        )

        # ----------------------------------------------------
        # Route score
        # ----------------------------------------------------

        metadata = _metadata(
            candidate
        )

        candidate_route = _get_route(
            metadata
        )

        route_score = 0.0

        if (
            query_route != "general"
            and candidate_route
            == query_route
        ):

            route_score = 6.0

        elif (
            query_route != "general"
            and candidate_route
            and candidate_route
            != query_route
        ):

            route_score = -(
                PENALTY_WRONG_ROUTE
            )

        legal_score += (
            route_score
            * 0.50
        )

        # ----------------------------------------------------
        # Intent-specific bonuses
        # ----------------------------------------------------

        document = _normalize_text(
            candidate.get(
                "document",
                "",
            )
        )

        title = _normalize_text(
            _get_section_title(
                metadata
            )
        )

        if query_intent == "punishment":

            if (
                "shall be punished"
                in document
                or "punished with"
                in document
            ):

                legal_score += (
                    BONUS_PUNISHMENT_PHRASE
                )

        elif (
            query_intent
            == "definition"
        ):

            if (
                "is said to"
                in document
                or "means"
                in document
                or "defined as"
                in document
            ):

                legal_score += (
                    BONUS_DEFINITION_PHRASE
                )

        # ----------------------------------------------------
        # Direct title match
        # ----------------------------------------------------

        if query_term:

            if query_term in title:

                legal_score += (
                    BONUS_DIRECT_TITLE
                )

        # Penalize amendment/footnote fragments
        if any(
            bad_phrase in title.lower() or bad_phrase in document[:200].lower()
            for bad_phrase in ["subs. by the a.o.", "subs. by act", "ins. by act", "repealed by", "a.o. 1950", "act 24 of 1917"]
        ):
            legal_score -= 25.0

        candidate[
            "topic_score"
        ] = float(
            topic_score
        )

        candidate[
            "route_score"
        ] = float(
            route_score
        )

        candidate[
            "legal_rerank_score"
        ] = float(
            legal_score
        )

        candidate[
            "query_intent"
        ] = query_intent

        candidate[
            "query_legal_term"
        ] = query_term

        candidate[
            "query_route"
        ] = query_route

        candidate[
            "section_expansion"
        ] = False

    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda item: float(
            item.get(
                "legal_rerank_score",
                0.0,
            )
        ),
        reverse=True,
    )

    # ========================================================
    # DIRECT SECTION EXPANSION
    # ========================================================

    direct_sections = []

    if query_term:

        candidate_sections = []

        for candidate in candidates:

            metadata = _metadata(
                candidate
            )

            section = _get_section(
                metadata
            )

            title = _normalize_text(
                _get_section_title(
                    metadata
                )
            )

            if not section:
                continue

            # Only expand sections from the expected route.
            if (
                query_route
                != "general"
                and _get_route(
                    metadata
                )
                != query_route
            ):

                continue

            is_direct_title_match = False

            configuration = (
                TOPIC_CONFIG.get(
                    query_term,
                    {},
                )
            )

            for term in configuration.get(
                "terms",
                [],
            ):

                if (
                    _normalize_text(
                        term
                    )
                    in title
                ):

                    is_direct_title_match = True
                    break

            if is_direct_title_match:

                if section not in candidate_sections:

                    candidate_sections.append(
                        section
                    )

        for section in candidate_sections[
            :SECTION_EXPANSION_K
        ]:

            expanded = (
                _find_section_documents(
                    section,
                    documents,
                    metadatas,
                )
            )

            for result in expanded:

                metadata = _metadata(
                    result
                )

                if (
                    query_route
                    != "general"
                    and _get_route(
                        metadata
                    )
                    != query_route
                ):

                    continue

                result[
                    "query_intent"
                ] = query_intent

                result[
                    "query_legal_term"
                ] = query_term

                result[
                    "query_route"
                ] = query_route

                result[
                    "legal_rerank_score"
                ] = (
                    BONUS_SECTION_EXPANSION
                )

                direct_sections.append(
                    result
                )

    # Add expansion results only after direct candidates.
    candidates.extend(
        direct_sections
    )

    # ========================================================
    # SECTION DEDUPLICATION
    # ========================================================

    candidates = (
        _deduplicate_sections(
            candidates
        )
    )

    # ========================================================
    # FINAL ROUTE SAFETY
    # ========================================================

    final_candidates = []

    for candidate in candidates:

        metadata = _metadata(
            candidate
        )

        candidate_route = _get_route(
            metadata
        )

        # For explicit legal topics, do not return a
        # completely unrelated legal route.
        if (
            query_route != "general"
            and candidate_route
            and candidate_route
            != query_route
        ):

            continue

        final_candidates.append(
            candidate
        )

        if len(
            final_candidates
        ) >= final_k:

            break

    # ========================================================
    # FALLBACK
    # ========================================================

    if not final_candidates:

        # If the route is general, return top candidates.
        if query_route == "general":

            final_candidates = candidates[
                :final_k
            ]

        else:

            # There were no route-matched results.
            # Return nothing rather than injecting unrelated law.
            final_candidates = []

    # ========================================================
    # HYBRID SCORE
    # ========================================================

    for candidate in final_candidates:

        candidate[
            "hybrid_score"
        ] = float(
            candidate.get(
                "legal_rerank_score",
                candidate.get(
                    "rrf_score",
                    0.0,
                ),
            )
        )

        candidate[
            "methods"
        ] = _json_safe(
            candidate.get(
                "methods",
                [],
            )
        )

    # ========================================================
    # JSON SAFE
    # ========================================================

    final_candidates = [
        _json_safe(
            candidate
        )
        for candidate in final_candidates
    ]

    print(
        f"Accepted legal sources: "
        f"{len(final_candidates)}"
    )

    return final_candidates


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main() -> None:

    import json

    test_queries = [
        "What is the punishment for theft?",
        "What is theft?",
        "What is the punishment for murder?",
        "What is cheating?",
        "loan repayment outstanding debt recovery",
        "How can an outstanding loan be recovered?",
        "property ownership dispute",
    ]

    print()
    print(
        "JanNyaya AI - Hybrid Legal Retriever Test"
    )

    for query in test_queries:

        print()
        print(
            "=" * 70
        )

        print(
            "QUERY:",
            query,
        )

        results = hybrid_search(
            query,
            semantic_k=20,
            bm25_k=20,
            final_k=5,
        )

        print()

        for index, result in enumerate(
            results,
            start=1,
        ):

            metadata = _metadata(
                result
            )

            print(
                json.dumps(
                    {
                        "rank":
                            index,

                        "section":
                            _get_section(
                                metadata
                            ),

                        "title":
                            _get_section_title(
                                metadata
                            ),

                        "source":
                            metadata.get(
                                "source",
                                "",
                            ),

                        "act":
                            metadata.get(
                                "title",
                                "",
                            ),

                        "route":
                            metadata.get(
                                "route",
                                "",
                            ),

                        "query_term":
                            result.get(
                                "query_legal_term",
                                "",
                            ),

                        "query_route":
                            result.get(
                                "query_route",
                                "",
                            ),

                        "score":
                            result.get(
                                "legal_rerank_score",
                                0.0,
                            ),
                    },
                    ensure_ascii=False,
                )
            )

    print()
    print(
        "Retriever test completed."
    )


if __name__ == "__main__":
    main()