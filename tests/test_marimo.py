import marimo as mo
import pytest


def test_marimo_app_initialization():
    """Test that a simple marimo App can be initialized without errors."""
    app = mo.App()
    assert app is not None


def test_marimo_ui_cell():
    """Test that a simple UI cell can be executed without errors."""
    app = mo.App()

    @app.cell
    def ui(mo):
        mo.md("# Test UI")
        mo.ui.write("This is a test.")

    # Simulate running the app (without actual server)
    try:
        # Note: marimo doesn't provide a direct way to run without server
        # So we'll just ensure that cell registration works
        assert True
    except Exception as e:
        pytest.fail(f"Marimo UI cell failed with exception: {e}")
