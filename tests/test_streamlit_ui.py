import pytest

from app.utils import FIELD_DISPLAY_CONFIG, display_field, display_response


class MockResponse:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestDisplayUtils:
    @pytest.fixture
    def output_list(self):
        return []

    @pytest.fixture
    def mock_writer(self, output_list):
        def writer(text):
            output_list.append(str(text))

        return writer

    def test_display_field_simple(self, output_list, mock_writer):
        display_field("response", "Test response", mock_writer)
        assert output_list == ["Response", "Test response"]

    def test_display_field_list(self, output_list, mock_writer):
        display_field("sources", ["source1", "source2"], mock_writer)
        assert output_list == ["Sources", "- source1", "- source2"]

    def test_display_field_confidence(self, output_list, mock_writer):
        display_field("confidence", 0.756, mock_writer)
        assert output_list == ["Confidence", "75.6%"]

    def test_display_field_empty(self, output_list, mock_writer):
        display_field("sources", [], mock_writer)
        assert output_list == []

    def test_display_field_unknown(self, output_list, mock_writer):
        display_field("unknown_field", "test value", mock_writer)
        assert output_list == ["Unknown Field", "test value"]

    def test_display_response(self, output_list, mock_writer):
        mock_response = MockResponse(
            response="Main response",
            confidence=0.85,
            sources=["src1", "src2"],
            evidence=["ev1", "ev2"],
        )

        display_response(mock_response, mock_writer)

        # Check each section is present but don't enforce order
        sections = {
            "Response": ["Response", "Main response"],
            "Confidence": ["Confidence", "85.0%"],
            "Sources": ["Sources", "- src1", "- src2"],
            "Evidence": ["Supporting Evidence", "- ev1", "- ev2"],
        }

        for section_items in sections.values():
            indices = [output_list.index(item) for item in section_items]
            # Not necessarliy consecutive
            assert indices == sorted(
                indices
            ), f"Items in section {section_items[0]} are not in correct order"

        # Verify all expected content is present
        expected_items = [item for section in sections.values() for item in section]
        assert all(
            item in output_list for item in expected_items
        ), "Not all expected items are present"
        assert len(output_list) == len(
            expected_items
        ), "Output contains unexpected items"

    def test_field_display_config_completeness(self):
        required_keys = {"title", "display_format", "is_list"}
        for field, config in FIELD_DISPLAY_CONFIG.items():
            assert all(
                key in config for key in required_keys
            ), f"Field {field} missing required configuration keys"
