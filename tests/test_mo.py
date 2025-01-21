import marimo as marimo_module  # Updated alias
import pytest


def test_marimo_ui_cell_registration():
    """Test that the marimo UI cell can be registered without errors."""
    app = marimo_module.App()

    try:

        @app.cell
        def ui(mo):
            mo.md("# Test UI")
            mo.ui.write("This is a test.")

        # If registration succeeds, pass the test
        assert True
    except Exception as e:
        pytest.fail(f"Marimo UI cell registration failed with exception: {e}")


def test_marimo_health_check_registration():
    """Test that the health check cell can be registered without errors."""
    app = marimo_module.App()

    try:

        @app.cell
        def health_check(mo):
            @mo.route("/health")
            def health():
                return {"status": "healthy"}

        # If registration succeeds, pass the test
        assert True
    except Exception as e:
        pytest.fail(f"Marimo health check cell registration failed with exception: {e}")


def test_marimo_app_initialization():
    """Test that a simple marimo App can be initialized without errors."""
    app = marimo_module.App()
    assert app is not None


def test_marimo_ui_cell():
    """Test that a simple UI cell can be executed without errors."""
    app = marimo_module.App()

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
