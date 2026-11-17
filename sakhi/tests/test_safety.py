"""Tests for pre-model safety overrides and privacy-sensitive app behavior."""

import json
from pathlib import Path

from sakhi.src import pipeline
from sakhi.src.pipeline import analyze
from sakhi.src.pipeline import _retrieve
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


def test_pipeline_retrieval_uses_configured_retriever_and_score_threshold(monkeypatch) -> None:
    seen = {}

    class FakeRetrieval:
        def retrieve(self, text, k, retriever):
            seen.update(text=text, k=k, retriever=retriever)
            return [
                {
                    "chunk_id": "FAKE-1",
                    "act": "FAKE LAW TEXT",
                    "section_no": "1",
                    "text": "1. FAKE LAW TEXT",
                    "score": 0.2,
                }
            ]

    monkeypatch.setattr("sakhi.src.pipeline.importlib.import_module", lambda _: FakeRetrieval())
    monkeypatch.setenv("SAKHI_RETRIEVER", "bm25_keywords")
    monkeypatch.setenv("SAKHI_RETRIEVAL_MIN_SCORE", "0.5")
    provisions, state = _retrieve("fake query", 3)
    assert provisions == []
    assert "no matching provision" in state
    assert seen == {"text": "fake query", "k": 3, "retriever": "bm25_keywords"}


def test_app_does_not_explicitly_store_message() -> None:
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "st.session_state" not in source
    assert 'key="msg_' not in source


def test_app_backend_returns_analyzer_when_models_are_available(monkeypatch) -> None:
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    source = app_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    if_line = next(line for line in lines if "if not mod.models_available():" in line)
    assert 'return getattr(mod, "analyze"), f"sakhi.src.pipeline.analyze ({mod.model_family()})"' in source
    assert len(if_line) - len(if_line.lstrip()) == 4


def test_pipeline_uses_available_embedding_model_for_queries(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models_embedding"
    model_dir.mkdir()
    metadata = {
        "model_family": "embedding",
        "model_name": "fake-model",
        "situation_labels": ["other", "stalking"],
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata))
    (model_dir / "situation_thresholds.json").write_text('{"other": 0.5, "stalking": 0.5}')
    for name in ("urgency_model.joblib", "situation_model.joblib"):
        (model_dir / name).touch()

    baseline_dir = tmp_path / "models"
    monkeypatch.setattr(pipeline, "MODELS_DIR", baseline_dir)
    monkeypatch.setattr(pipeline, "EMBEDDING_MODELS_DIR", model_dir)
    monkeypatch.delenv("SAKHI_MODELS_DIR", raising=False)
    assert pipeline.models_dir() == model_dir
    assert pipeline.models_available()

    class FakeEncoder:
        inputs = None

        def encode(self, texts, show_progress_bar=False):
            self.inputs = texts
            return [[0.1, 0.2]]

    class FakeUrgencyModel:
        classes_ = ["urgent", "non_urgent"]

        def predict_proba(self, features):
            return [[0.8, 0.2]]

    class FakeSituationModel:
        def predict_proba(self, features):
            return [[0.9, 0.1]]

    encoder = FakeEncoder()
    loaded = {"urgency_model.joblib": FakeUrgencyModel(),
              "situation_model.joblib": FakeSituationModel()}
    monkeypatch.setattr(pipeline.joblib, "load", lambda path: loaded[Path(path).name])
    monkeypatch.setattr(pipeline, "_load_encoder", lambda name: encoder)
    monkeypatch.setattr(pipeline, "_retrieve", lambda text, k: ([], "not connected"))
    pipeline.load_artifacts.cache_clear()

    result = pipeline.analyze("Can you help me understand my options?")

    assert encoder.inputs and len(encoder.inputs) == 1
    assert result["urgency"]["label"] == "urgent"
    assert result["situations"][0]["label"] == "other"
