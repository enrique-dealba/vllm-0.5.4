from unittest.mock import MagicMock

import pytest


class MockSettings:
    USE_STRUCTURED_OUTPUT = True


mock_settings = MockSettings()


@pytest.fixture(autouse=True)
def mock_imports(monkeypatch):
    monkeypatch.setattr("app.streamlit_ui.settings", mock_settings)
    monkeypatch.setattr(
        "app.streamlit_ui.generate_response",
        lambda x: (MagicMock(response="test response"), 0.1),
    )
