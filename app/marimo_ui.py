import logging

import marimo

from app.config import settings
from app.llm_logic import generate_response
from app.utils import get_displayable_fields

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = marimo.App()

# Expose the logger via marimo
marimo.logger = logger


@app.cell
def ui(context):
    context.logger.info("Rendering UI")
    context.md("# LLM")

    with context.form("query_form", submit_label="Generate"):
        user_input = context.ui.text_input("Enter your query:")
        submit = context.ui.submit_button()

    if submit:
        if not user_input:
            context.ui.warning("Please enter a query.")
        else:
            try:
                llm_response, execution_time = generate_response(user_input)
                if settings.USE_STRUCTURED_OUTPUT:
                    fields = get_displayable_fields(llm_response)
                    context.ui.json(fields)
                    context.ui.info(f"Execution time: {execution_time:.2f} seconds")
                else:
                    context.ui.write(llm_response)
            except Exception as e:
                context.logger.error(f"An error occurred: {e}")
                context.ui.error(f"An error occurred: {e}")


@app.cell
def health_check(context):
    @context.route("/health")
    def health():
        return {"status": "healthy"}


@app.cell
def diagnostic(context):
    if context is None:
        print("Diagnostic: context is None")
    else:
        context.logger.info("Diagnostic: context is defined")
        context.md("# Diagnostic")
        context.ui.write("Context is properly passed to cell functions.")


if __name__ == "__main__":
    logger.info(f"Starting marimo app on port {settings.PORT}")
    app.run(host="0.0.0.0", port=settings.PORT)
