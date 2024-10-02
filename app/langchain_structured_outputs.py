import time

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from app.model import llm
from app.utils import load_schema


def generate_response(user_input: str):
    start_time = time.time()

    try:
        LLMResponseSchema = load_schema()
        parser = PydanticOutputParser(pydantic_object=LLMResponseSchema)

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser

        response = chain.invoke({"query": user_input})

        end_time = time.time()
        execution_time = end_time - start_time

        return response, execution_time

    except Exception as e:
        print(f"Error during structured output generation: {e}")
        raise
