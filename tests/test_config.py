import utils.config as config


def test_default_min_retrieval_score_when_unset(monkeypatch):
    monkeypatch.delenv("MIN_RETRIEVAL_SCORE", raising=False)
    assert config.get_min_retrieval_score() == 0.35


def test_reads_valid_value_from_env(monkeypatch):
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.6")
    assert config.get_min_retrieval_score() == 0.6


def test_invalid_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "not-a-number")
    assert config.get_min_retrieval_score() == 0.35


def test_out_of_range_value_is_clamped(monkeypatch):
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "5.0")
    assert config.get_min_retrieval_score() == 1.0
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "-2.0")
    assert config.get_min_retrieval_score() == 0.0


def test_custom_default_is_respected_when_env_unset(monkeypatch):
    monkeypatch.delenv("MIN_RETRIEVAL_SCORE", raising=False)
    assert config.get_min_retrieval_score(default=0.5) == 0.5
