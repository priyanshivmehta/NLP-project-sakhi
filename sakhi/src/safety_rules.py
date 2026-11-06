"""Configuration-driven safety overrides for high-risk messages."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = PROJECT_ROOT / "sakhi" / "config" / "safety_rules.yaml"
HELPLINES_PATH = PROJECT_ROOT / "sakhi" / "config" / "helplines.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    with RULES_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Safety rules must be a mapping: {RULES_PATH}")
    return config


@lru_cache(maxsize=1)
def _load_helplines() -> dict[str, Any]:
    with HELPLINES_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("helplines", {})


def _matches(text: str, pattern_names: list[str], patterns: dict[str, list[str]]) -> bool:
    lowered = text.casefold()
    return any(
        re.search(pattern, lowered, flags=re.IGNORECASE | re.UNICODE)
        for name in pattern_names
        for pattern in patterns.get(name, [])
    )


def _contact(key: str) -> dict[str, Any]:
    contact = dict(_load_helplines().get(key, {}))
    contact["key"] = key
    return contact


def evaluate_safety(text: str) -> dict[str, Any]:
    """Return safety overrides before any model or legal retrieval is run."""
    config = _load_config()
    patterns = config.get("patterns", {})
    rules = config.get("rules", {})
    immediate = rules.get("immediate_danger", {})
    danger = _matches(text, immediate.get("any", []), patterns)
    threat = _matches(text, immediate.get("threat_with_now", []), patterns)
    now = _matches(text, ["now"], patterns)
    immediate = danger or (threat and now)
    minor = _matches(text, rules.get("minor", {}).get("any", []), patterns)
    self_harm = _matches(text, rules.get("self_harm", {}).get("any", []), patterns)
    contacts = [_contact("emergency")] if immediate else []
    if minor:
        contacts.append(_contact("child_protection"))
    if self_harm:
        contacts.append(_contact("crisis_support"))
    return {
        "immediate_danger": immediate,
        "minor": minor,
        "self_harm": self_harm,
        "contacts": contacts,
        "caring_message": config.get("messages", {}).get("self_harm_caring") if self_harm else None,
        "skip_legal_explanation": self_harm,
        "rule_fired": immediate or minor or self_harm,
    }
