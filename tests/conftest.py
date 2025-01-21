import pytest


@pytest.fixture(autouse=True)
def mock_imports(monkeypatch):
    class MockSettings:
        USE_STRUCTURED_OUTPUT = True

    class MockConfig:
        settings = MockSettings()

    class MockLLMLogic:
        def generate_response(self, input_text):
            return "Mock response", 0.0

    monkeypatch.setattr("app.streamlit_ui.settings", MockSettings())
    monkeypatch.setattr(
        "app.streamlit_ui.generate_response", MockLLMLogic().generate_response
    )
