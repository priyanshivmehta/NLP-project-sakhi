"""Tests for the five pure preprocessing stages."""

from pathlib import Path

import pandas as pd

from sakhi.src.preprocess import (
    clean_text,
    mask_entities,
    normalize_text,
    preprocess_dataframe,
    raw_stage,
)


def test_raw_stage_is_unchanged() -> None:
    text = "  नमस्ते  🙂 "
    assert raw_stage(text) == text


def test_cleaning_preserves_devanagari_and_marks_url_and_emoji() -> None:
    result = clean_text("नमस्ते   https://example.test 🙂")
    assert "नमस्ते" in result
    assert "[URL]" in result
    assert "[EMOJI]" in result
    assert "https://" not in result


def test_normalisation_handles_hinglish_and_repeated_characters() -> None:
    result = normalize_text("Pleaseee pls pareshaan", "hinglish", {"pls": "please", "pareshaan": "परेशान"})
    assert result == "pleasee please परेशान"


def test_normalisation_does_not_lowercase_devanagari_text() -> None:
    text = "यह एक उदाहरण है"
    assert normalize_text(text, "hi") == text


def test_entity_masking_handles_mixed_text() -> None:
    result = mask_entities("Call 9876543210 or email Sakhi.Test@example.org. Ref 123456789.")
    assert "[PHONE]" in result
    assert "[EMAIL]" in result
    assert "[LONG_NUMBER]" in result
    assert "9876543210" not in result


def test_entity_masking_preserves_emergency_helpline_numbers() -> None:
    result = mask_entities("Please call 112 or women helpline 181.")
    assert "112" in result
    assert "181" in result
    assert "[PHONE]" not in result


def test_entity_masking_handles_common_phone_formats() -> None:
    result = mask_entities("+91 98765 43210; 98765-43210; 9876543210")
    assert result.count("[PHONE]") == 3
    assert "98765" not in result
    assert "43210" not in result


def test_entity_masking_masks_twelve_digit_numbers() -> None:
    result = mask_entities("Reference 123456789012")
    assert result == "Reference [LONG_NUMBER]"


def test_entity_masking_does_not_mask_names_yet() -> None:
    result = mask_entities("Rahul keeps following me.")
    assert "Rahul" in result


def test_dataframe_snapshot_contains_every_stage(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "text": ["नमस्ते 🙂", "Mujheee help pls", "Mixed नमस्ते 123456"],
            "language": ["hi", "hinglish", "en"],
        }
    )
    output = tmp_path / "snapshot.csv"
    snapshot = preprocess_dataframe(data, output_path=output)
    assert output.exists()
    assert list(snapshot.columns[-5:]) == [
        "raw",
        "cleaning",
        "normalisation",
        "entity_masking",
        "final_model_input",
    ]
    assert snapshot.loc[0, "raw"] == "नमस्ते 🙂"
    assert "[EMOJI]" in snapshot.loc[0, "cleaning"]
    assert "123456" not in snapshot.loc[2, "entity_masking"]
