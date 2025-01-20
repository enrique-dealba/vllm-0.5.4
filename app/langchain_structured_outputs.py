from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from app.model import llm
from app.utils import load_schema, time_function


@time_function
def generate_structured_response(user_input: str):
    LLMResponseSchema = load_schema()

    parser = PydanticOutputParser(pydantic_object=LLMResponseSchema)

    template = """You are a helpful AI assistant that always responds in valid JSON format.
    
Format your response according to this schema:
{format_instructions}

Remember:
1. Your response MUST be valid JSON
2. Do not include any explanatory text outside the JSON
3. Ensure all required fields are included
4. Use the exact field names specified

User Query: {query}

JSON Response:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        return chain.invoke({"query": user_input})
    except Exception:
        # Fallback to basic response
        basic_response = LLMResponseSchema(
            response=f"I apologize, but I encountered an error processing your request. Here's my response: {llm.invoke(user_input)}",
            confidence=0.5,
        )
        return basic_response
