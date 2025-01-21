import marimo as mo

from app.config import settings
from app.llm_logic import generate_response
from app.utils import get_displayable_fields

app = mo.App(title="LLM")


@app.cell
def ui():
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
                mo.ui.error(f"An error occurred: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.PORT)
