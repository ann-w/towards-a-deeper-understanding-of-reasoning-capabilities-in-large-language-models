# Replace with your own LLM API.
# Note: query_model takes two arguments: 1) message in openai chat completion form (list of dictionaries),
#                                        2) an index to indicate where the message should be truncated if the length exceeds LLM context length.

import os
from typing import Callable, Dict, List, Union

# Added by Annie
import dotenv
from openai import OpenAI
from transformers import pipeline

# Originally from libs/smartplay/examples/llm_api.py
# def get_query(LLM_name):
#     # Initialize the pipeline
#     generator = pipeline("text-generation", model=LLM_name)

#     def query_model(message, index):
#         # Truncate the message if necessary
#         if len(message) > index:
#             message = message[:index]

#         # Convert the message list to a single string
#         message_str = " ".join([msg["content"] for msg in message])

#         # Query the LLM
#         response = generator(message_str, max_new_tokens=50)

#         # Return the assistant's response
#         return response[0]["generated_text"]

#     return query_model


# Added by Annie
def get_query(model_name: str) -> Callable:
    # Initialize the OpenAI client with the API key
    dotenv.load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    def query_model(messages: List[Dict[str, str]], index: int) -> Union[str, int, int]:

        # Truncate the message if necessary
        if len(messages) > index:
            messages = messages[:index]

        # Create a completion request to OpenAI chat API
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=150,
            # temperature=temp_value,
            n=1,  # Generate one completion per request
        )

        # Extract the assistant's response content
        output_text = response.choices[0].message.content

        # Return the assistant's response
        return output_text

    return query_model
