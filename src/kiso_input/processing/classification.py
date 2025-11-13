"""Self-harm classification utilities based on lexicon heuristics."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .segments import FREE_TEXT_TOKENS


def _lazy_load_spacy():
    try:
        import spacy  # type: ignore

        try:
            return spacy.load("de_core_news_sm")
        except Exception:
            return spacy.blank("de")
    except Exception:
        return None


def _normalize_text(text: str, normalize_cfg: Dict[str, Any]) -> str:
    t = unicodedata.normalize("NFKC", text)
    if normalize_cfg.get("lower", True):
        t = t.lower()
    if normalize_cfg.get("umlaut_normalization", True):
        t = (
            t.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )
    if normalize_cfg.get("strip_punct", True):
        t = re.sub(r"[^\w\s„“\"’'…\-.,;:!?()\[\]]+", " ", t)
    if normalize_cfg.get("collapse_whitespace", True):
        t = re.sub(r"\s+", " ", t).strip()
    return t


def _split_sentences(text: str, nlp=None) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Return list of (sentence_text, (start_idx, end_idx)) in normalised text.
    Uses spaCy if available, otherwise falls back to a regex heuristic.
    """
    if nlp:
        doc = nlp(text)
        if doc.has_annotation("SENT_START"):
            return [(sent.text, (sent.start_char, sent.end_char)) for sent in doc.sents]

    spans: List[Tuple[str, Tuple[int, int]]] = []
    start = 0
    for match in re.finditer(r"([^.!?]+[.!?])", text):
        spans.append((match.group(0).strip(), (match.start(), match.end())))
        start = match.end()
    if start < len(text):
        spans.append((text[start:].strip(), (start, len(text))))
    return spans


def _in_window(haystack: str, pivot_idx: int, window: int, patterns: List[str]) -> bool:
    lo = max(0, pivot_idx - window)
    hi = min(len(haystack), pivot_idx + window)
    segment = haystack[lo:hi]
    for pattern in patterns:
        if re.search(rf"\b{re.escape(pattern)}\b", segment):
            return True
    return False


def _has_reported_speech(sentence: str, markers: List[str]) -> bool:
    """Simple heuristic: detect quotation marks or marker words."""
    if any(symbol in sentence for symbol in ["„", "“", '"', "»", "«"]):
        return True
    for marker in markers:
        if re.search(rf"\b{re.escape(marker)}\b", sentence):
            return True
    return False


def _deobfuscate_text(text: str) -> str:
    """Remove spaces and punctuation between letters to handle obfuscation like 'u.m.b.r.i.n.g.e.n'."""
    # Replace sequences of letter + (space/punctuation) + letter with just letters
    # This handles patterns like "u.m.b.r.i.n.g.e.n" or "u m b r i n g e n"
    # Keep doing it until no more changes (handles multiple separators)
    prev = text
    while True:
        deobfuscated = re.sub(r"([a-zäöüß])\s*[^\w\s]*\s*([a-zäöüß])", r"\1\2", prev, flags=re.IGNORECASE)
        if deobfuscated == prev:
            break
        prev = deobfuscated
    return deobfuscated


def _create_flexible_pattern(phrase: str) -> str:
    """Create a flexible regex pattern that allows optional words between 'mich' and 'um'."""
    # For phrases like "ich bringe mich um", allow optional words between "mich" and "um"
    phrase_lower = phrase.lower()
    if " mich " in phrase_lower and " um" in phrase_lower:
        # Split at "mich" to get parts before and after
        parts = re.split(r"\s+mich\s+", phrase, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) == 2:
            before_mich = re.escape(parts[0].strip())
            after_mich = parts[1].strip()
            # Check if "um" appears after "mich"
            if after_mich.lower().startswith("um"):
                # Allow 0-3 words between "mich" and "um"
                # Pattern: before_mich + " mich " + (0-3 words) + " um"
                pattern = rf"\b{before_mich}\s+mich\s+(\w+\s+){{0,3}}um\b"
                return pattern
    return re.escape(phrase)


def _find_all_occurrences(sentence: str, phrases: List[str]) -> List[int]:
    """Find all occurrences of phrases in sentence, handling obfuscation."""
    indices: List[int] = []
    deobfuscated_sentence = _deobfuscate_text(sentence)
    
    for phrase in phrases:
        # Escape special regex characters but allow flexible matching
        escaped_phrase = re.escape(phrase)
        
        # Try exact match first (case-insensitive)
        for match in re.finditer(escaped_phrase, sentence, re.IGNORECASE):
            indices.append(match.start())
        
        # Try flexible pattern for "mich ... um" constructions
        flexible_pattern = _create_flexible_pattern(phrase)
        if flexible_pattern != escaped_phrase:
            for match in re.finditer(flexible_pattern, sentence, re.IGNORECASE):
                indices.append(match.start())
        
        # Also try deobfuscated version if different
        deobfuscated_phrase = _deobfuscate_text(phrase)
        if deobfuscated_phrase != phrase and len(deobfuscated_phrase) >= 3:  # Only for substantial phrases
            # Search in deobfuscated sentence
            escaped_deobfuscated = re.escape(deobfuscated_phrase)
            for match in re.finditer(escaped_deobfuscated, deobfuscated_sentence, re.IGNORECASE):
                indices.append(match.start())
    
    return sorted(set(indices))  # Remove duplicates and sort


def classify_self_harm(text: str, lexicon: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply lexicon-based heuristics to classify self-harm risk in free-text answers.
    """
    normalize_cfg = (lexicon.get("defaults", {}) or {}).get("normalize", {}) or {}
    window_chars = (lexicon.get("defaults", {}) or {}).get("window_chars", 80)
    base_weight = (lexicon.get("defaults", {}) or {}).get("weight", 1.0)
    categories = {c["id"]: c for c in lexicon.get("categories", [])}
    signals = lexicon.get("signals", [])
    aux = lexicon.get("aux", {})
    first_person = aux.get("first_person_pronouns", [])
    negations = aux.get("negations", [])
    imminent = aux.get("imminent_time", [])
    reported_markers = aux.get("reported_speech_markers", [])
    means_lex = set(aux.get("means_lexicon", []))

    nlp = _lazy_load_spacy()
    norm_text = _normalize_text(text, normalize_cfg)
    sentences = _split_sentences(norm_text, nlp=nlp)

    category_scores = {c_id: 0.0 for c_id in categories.keys()}
    fired_signals: List[Dict[str, Any]] = []
    applied_rules: List[str] = []

    for sent_text, (s_start, s_end) in sentences:
        is_reported = _has_reported_speech(sent_text, reported_markers)
        for signal in signals:
            phrases = [signal["phrase"]] + signal.get("variants", [])
            hit_positions = _find_all_occurrences(sent_text, phrases)
            if not hit_positions:
                continue
            category = signal["category"]
            weight = signal.get("weight", base_weight)
            severity = categories.get(category, {}).get("severity", 0)
            rule_notes: List[str] = []

            for pos in hit_positions:
                if signal.get("context_hints", {}).get("first_person_required", False):
                    has_fp = _in_window(sent_text, pos, window_chars, first_person)
                    if not has_fp:
                        rule_notes.append("no_first_person_downgrade")
                        weight *= 0.6

                if signal.get("negation_overrides"):
                    if _in_window(sent_text, pos, window_chars, signal["negation_overrides"].get("patterns", [])):
                        effect = signal["negation_overrides"].get("effect", "downgrade")
                        rule_notes.append(f"negation_{effect}")
                        if effect == "downgrade":
                            weight *= 0.5

                if _in_window(sent_text, pos, window_chars, imminent):
                    rule_notes.append("imminent_context")
                    weight *= 1.2

                if signal.get("context_hints", {}).get("means"):
                    means_here = any(
                        _in_window(sent_text, pos, window_chars, [m])
                        for m in signal["context_hints"]["means"]
                    )
                    if means_here:
                        rule_notes.append("means_present")
                        weight *= 1.2
                elif any(_in_window(sent_text, pos, window_chars, [m]) for m in means_lex):
                    rule_notes.append("means_present_global")
                    weight *= 1.1

                is_context = False
                if signal.get("context_hints", {}).get("reported_speech", False) or is_reported:
                    if not _in_window(sent_text, pos, window_chars, first_person):
                        is_context = True
                        rule_notes.append("reported_speech_context")
                        weight *= 0.5
                        category = "drittperson_oder_kontext"
                        severity = categories.get(category, {}).get("severity", 0)

                if signal.get("counter_signals"):
                    if _in_window(sent_text, pos, window_chars, signal["counter_signals"].get("patterns", [])):
                        effect = signal["counter_signals"].get("effect", "confirm_nssi")
                        rule_notes.append(f"counter_{effect}")
                        if effect == "confirm_nssi":
                            category = "svv_nicht_suizidal"
                            severity = categories.get(category, {}).get("severity", 0)

                if signal.get("escalation_rules", {}).get("set_risk") == "high":
                    rule_notes.append("rule_set_risk_high")

                category_scores[category] = category_scores.get(category, 0.0) + weight
                fired_signals.append(
                    {
                        "signal_id": signal.get("id"),
                        "category": category,
                        "severity": severity,
                        "weight_applied": round(weight, 3),
                        "sentence_span": (s_start, s_end),
                        "position_in_sentence": pos,
                        "notes": rule_notes,
                        "is_context": is_context,
                    }
                )

    safe_score = category_scores.get("erholung_oder_sicherheit", 0.0)
    if safe_score > 0:
        for key in ["suizid", "svv"]:
            if category_scores.get(key, 0) > 0:
                category_scores[key] *= 0.85
        applied_rules.append("deescalate_by_safety_signals")

    sorted_labels = sorted(
        [(key, value) for key, value in category_scores.items() if value > 0],
        key=lambda item: item[1],
        reverse=True,
    )

    risk_level = "niedrig"
    high_rule = any("rule_set_risk_high" in fs["notes"] for fs in fired_signals)
    if high_rule or category_scores.get("suizid", 0.0) > 0:
        risk_level = "hoch"
    elif category_scores.get("svv", 0.0) > 0:
        risk_level = "mittel"

    return {
        "risk_level": risk_level,
        "category_scores": {key: round(value, 3) for key, value in category_scores.items()},
        "fired_signals": fired_signals,
        "applied_rules": applied_rules,
        "final_labels": [key for key, _ in sorted_labels],
        "notes": "Heuristik + Lexikon; optional mit Modellwahrscheinlichkeiten kombinieren.",
    }


def load_self_harm_lexicon(path: str | Path) -> Dict[str, Any]:
    """Load the self-harm lexicon from YAML."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Lexicon not found: {target}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyYAML wird benötigt. Installiere es mit: pip install pyyaml") from exc

    with target.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Lexicon file must contain a dictionary at root: {target}")
    return data


def extract_free_text_answers(segments: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Extract (question, answer) tuples for all free-text answers present in segments."""
    results: List[Tuple[str, str]] = []
    current_question: Optional[str] = None
    for segment in segments:
        if "Question" in segment:
            current_question = segment.get("Question")
        elif "Answer" in segment and current_question:
            answer_val = segment.get("Answer")
            if isinstance(answer_val, str):
                lower = answer_val.strip().lower()
                if lower in FREE_TEXT_TOKENS:
                    continue
                if answer_val.strip():
                    results.append((current_question, answer_val.strip()))
    return results


def assess_free_text_answers(
    segments: List[Dict[str, Any]],
    lexicon: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run self-harm classification for each free-text answer in the given segments."""
    assessments: List[Dict[str, Any]] = []
    for question, answer in extract_free_text_answers(segments):
        assessments.append(
            {
                "question": question,
                "answer": answer,
                "analysis": classify_self_harm(answer, lexicon),
            }
        )
    return assessments


__all__ = [
    "classify_self_harm",
    "load_self_harm_lexicon",
    "extract_free_text_answers",
    "assess_free_text_answers",
]


