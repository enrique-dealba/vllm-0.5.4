from unittest.mock import patch

import pytest

from app.streamlit_ui import FIELD_DISPLAY_CONFIG, display_field, display_response


class MockResponse:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_streamlit():
    with patch("app.streamlit_ui.st") as mock_st:
        yield mock_st


def test_display_field_simple(mock_streamlit):
    """Test displaying a simple field"""
    display_field("response", "Test response")
    mock_streamlit.subheader.assert_called_once_with("Response")
    mock_streamlit.write.assert_called_once_with("Test response")


def test_display_field_list(mock_streamlit):
    """Test displaying a list field"""
    display_field("sources", ["source1", "source2"])
    mock_streamlit.subheader.assert_called_once_with("Sources")
    assert mock_streamlit.write.call_count == 2
    mock_streamlit.write.assert_any_call("- source1")
    mock_streamlit.write.assert_any_call("- source2")


def test_display_field_confidence(mock_streamlit):
    """Test displaying confidence field"""
    display_field("confidence", 0.756)
    mock_streamlit.subheader.assert_called_once_with("Confidence")
    mock_streamlit.write.assert_called_once_with("75.6%")


def test_display_response(mock_streamlit):
    """Test displaying full response object"""
    mock_response = MockResponse(
        response="Main response",
        confidence=0.85,
        sources=["src1", "src2"],
        evidence=["ev1", "ev2"],
    )

    display_response(mock_response)

    # Verify subheader calls
    expected_subheaders = ["Response", "Confidence", "Sources", "Supporting Evidence"]
    actual_subheader_calls = [
        call[0][0] for call in mock_streamlit.subheader.call_args_list
    ]
    assert actual_subheader_calls == expected_subheaders


def test_display_field_empty(mock_streamlit):
    """Test that empty fields are skipped"""
    display_field("sources", [])
    mock_streamlit.subheader.assert_not_called()
    mock_streamlit.write.assert_not_called()


def test_display_field_unknown(mock_streamlit):
    """Test handling of unknown field types"""
    display_field("unknown_field", "test value")
    mock_streamlit.subheader.assert_called_once_with("Unknown Field")
    mock_streamlit.write.assert_called_once_with("test value")


def test_field_display_config_completeness():
    """Test that all field configurations have required keys"""
    required_keys = {"title", "display_format", "is_list"}
    for field, config in FIELD_DISPLAY_CONFIG.items():
        assert all(
            key in config for key in required_keys
        ), f"Field {field} missing required configuration keys"
