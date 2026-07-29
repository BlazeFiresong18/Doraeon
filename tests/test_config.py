import core.config as config


def test_default_ollama_model_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert config.get_ollama_model() == "llama3.2"


def test_ollama_model_reads_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    assert config.get_ollama_model() == "mistral"


def test_default_ollama_base_url_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert config.get_ollama_base_url() == "http://localhost:11434"


def test_ollama_base_url_reads_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.50:11434")
    assert config.get_ollama_base_url() == "http://192.168.1.50:11434"


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
