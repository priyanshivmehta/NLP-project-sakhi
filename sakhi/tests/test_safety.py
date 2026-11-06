"""Tests for pre-model safety overrides and privacy-sensitive app behavior."""

from pathlib import Path

from sakhi.src.pipeline import analyze
from sakhi.src.safety_rules import evaluate_safety


IMMEDIATE_MESSAGES = [
    ("A weapon is here.", "Bandook hai.", "यहां हथियार है।"),
    ("There is acid outside.", "Tezab hai.", "यहां तेजाब है।"),
    ("He has a knife.", "Uske paas chaku hai.", "उसके पास चाकू है।"),
    ("I am locked in.", "Mujhe band karke rakha hai.", "मुझे बंद करके रखा है।"),
    ("I was kidnapped.", "Mujhe utha liya.", "मेरा अपहरण हुआ।"),
    ("There is a threat right now.", "Dhamki de raha hai abhi.", "अभी धमकी दे रहा है।"),
]


def test_immediate_danger_patterns_force_112_before_models() -> None:
    for messages in IMMEDIATE_MESSAGES:
        for message in messages:
            result = evaluate_safety(message)
            assert result["immediate_danger"] is True
            assert result["contacts"][0]["key"] == "emergency"


def test_minor_adds_child_protection_for_all_languages() -> None:
    for message in ("A child is involved.", "Bacchi ko dhamki hai.", "नाबालिग बच्ची"):
        result = evaluate_safety(message)
        assert result["minor"] is True
        assert any(contact["key"] == "child_protection" for contact in result["contacts"])


def test_self_harm_adds_crisis_support_and_skips_legal() -> None:
    for message in ("I want to kill myself.", "Mujhe khudkushi karni hai.", "मैं आत्महत्या करना चाहती हूं।"):
        result = evaluate_safety(message)
        assert result["self_harm"] is True
        assert result["caring_message"]
        assert result["skip_legal_explanation"] is True
        assert any(contact["key"] == "crisis_support" for contact in result["contacts"])


def test_pipeline_safety_override_does_not_load_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "sakhi.src.pipeline.load_artifacts",
        lambda: (_ for _ in ()).throw(AssertionError("model should not load")),
    )
    result = analyze("There is a knife here.")
    assert result["safety"]["immediate_danger"] is True
    assert result["provisions"] == []


def test_app_does_not_explicitly_store_message() -> None:
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "st.session_state" not in source
    assert 'key="msg_' not in source
