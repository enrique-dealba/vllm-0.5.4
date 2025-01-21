import logging

import marimo as marimo_module

from app.config import settings
from app.llm_logic import generate_response
from app.utils import get_displayable_fields

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = marimo_module.App()

# Expose logger via marimo_module
marimo_module.logger = logger


@app.cell
def ui(mo):
    mo.logger.info("Rendering UI")
    mo.md("# LLM")

    with mo.form("query_form", submit_label="Generate"):
        user_input = mo.ui.text_input("Enter your query:")
        submit = mo.ui.submit_button()

    if submit:
        if not user_input:
            mo.ui.warning("Please enter a query.")
        else:
            try:
                llm_response, execution_time = generate_response(user_input)
                if settings.USE_STRUCTURED_OUTPUT:
                    fields = get_displayable_fields(llm_response)
                    mo.ui.json(fields)
                    mo.ui.info(f"Execution time: {execution_time:.2f} seconds")
                else:
                    mo.ui.write(llm_response)
            except Exception as e:
                mo.logger.error(f"An error occurred: {e}")
                mo.ui.error(f"An error occurred: {e}")


@app.cell
def health_check(mo):
    @mo.route("/health")
    def health():
        return {"status": "healthy"}


if __name__ == "__main__":
    logger.info(f"Starting marimo app on port {settings.PORT}")
    app.run(host="0.0.0.0", port=settings.PORT)
