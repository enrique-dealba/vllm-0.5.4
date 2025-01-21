from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_streamlit():
    with patch("streamlit.st", MagicMock()) as mock_st:
        yield mock_st
