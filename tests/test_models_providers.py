from langclaw.models.providers import ModelConfig, ModelProvider


def test_model_provider_routes_by_prefix(monkeypatch):
    called = []

    def fake_openai(cfg):
        called.append(("openai", cfg.model_id))
        return "openai-model"

    def fake_anthropic(cfg):
        called.append(("anthropic", cfg.model_id))
        return "anthropic-model"

    def fake_google(cfg):
        called.append(("google", cfg.model_id))
        return "google-model"

    monkeypatch.setattr(ModelProvider, "get_openai", staticmethod(fake_openai))
    monkeypatch.setattr(ModelProvider, "get_anthropic", staticmethod(fake_anthropic))
    monkeypatch.setattr(ModelProvider, "get_google", staticmethod(fake_google))

    assert ModelProvider.get(ModelConfig(model_id="anthropic/claude-3-5-sonnet")) == "anthropic-model"
    assert ModelProvider.get(ModelConfig(model_id="google/gemini-1.5-pro")) == "google-model"
    assert ModelProvider.get(ModelConfig(model_id="openai/gpt-4o-mini")) == "openai-model"
    assert ModelProvider.get(ModelConfig(model_id="ollama/llama3.1")) == "openai-model"

    assert called == [
        ("anthropic", "anthropic/claude-3-5-sonnet"),
        ("google", "google/gemini-1.5-pro"),
        ("openai", "openai/gpt-4o-mini"),
        ("openai", "ollama/llama3.1"),
    ]
