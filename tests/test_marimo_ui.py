import marimo as mo
import pytest


def test_marimo_ui_cell_registration():
    """Test that the marimo UI cell can be registered without errors."""
    app = mo.App()

    try:

        @app.cell
        def ui(*, mo):
            mo.md("# Test UI")
            mo.ui.write("This is a test.")

        # If registration succeeds, pass the test
        assert True
    except Exception as e:
        pytest.fail(f"Marimo UI cell registration failed with exception: {e}")


def test_marimo_health_check_registration():
    """Test that the health check cell can be registered without errors."""
    app = mo.App()

    try:

        @app.cell
        def health_check(*, mo):
            @mo.route("/health")
            def health():
                return {"status": "healthy"}

        # If registration succeeds, pass the test
        assert True
    except Exception as e:
        pytest.fail(f"Marimo health check cell registration failed with exception: {e}")
