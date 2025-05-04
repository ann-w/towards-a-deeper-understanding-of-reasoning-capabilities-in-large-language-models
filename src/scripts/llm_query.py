import logging
import os
import time
from typing import Callable, Dict, List, Tuple

import dotenv
import ollama
import requests
import torch
import yaml
from google import genai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_config(model_name: str) -> Dict:
    model_name = model_name.split("/")[-1].lower()
    config_path = f"src/config/model_parameters/{model_name}.yaml"

    # Check if this is a Gemini model
    if model_name.startswith("gemini-"):
        # If config file exists, use it
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        else:
            # Return default config for Gemini models
            logging.info(
                f"No configuration file found for Gemini model: {model_name}. Using defaults."
            )
            return {
                "temperature": 0.2,
                "max_tokens": 1024,
                "top_p": 0.95,
                "do_sample": True,
            }

    # For non-Gemini models, require config file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No configuration file found for model: {model_name}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def count_tokens(text: str, tokenizer) -> int:
    """
    Counts the number of tokens in a given text using the specified tokenizer.

    Args:
        text (str): The input text to tokenize.
        tokenizer: The HuggingFace tokenizer to use.

    Returns:
        int: The number of tokens in the input text.
    """
    tokens = tokenizer.encode(text, return_tensors="pt")
    return len(tokens[0])


def get_deepseek_query(
    model_name: str,
    deepseek_api_key: str,
    config: Dict,
    count_token_usage: bool = True,
    temperature=None,
) -> Callable:
    """
    Creates a query function for DeepSeek's API, which is compatible with the OpenAI client.

    Args:
        model_name (str): The DeepSeek model to use (e.g., "deepseek-chat").
        deepseek_api_key (str): The API key for DeepSeek.
        config (Dict): Configuration parameters such as max_tokens, temperature, etc.
        count_token_usage (bool): Whether to include token usage in the return values.
        temperature: Temperature override; if None, will be read from config.

    Returns:
        Callable: A function that queries the DeepSeek model.
    """
    # Initialize the DeepSeek (OpenAI-compatible) client
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    def query_model(
        messages: List[Dict[str, str]],
        count_token_usage: bool = count_token_usage,
        temperature=temperature,
    ) -> Tuple[str, int, int]:

        # Determine temperature value
        temp_value = (
            temperature if temperature is not None else config.get("temperature", 0.01)
        )

        try:
            # Create a completion request to DeepSeek (OpenAI-compatible) chat API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=config.get("max_tokens", 150),
                temperature=temp_value,
                n=1,  # Generate one completion per request
                stream=False,
            )

            # Extract the assistant's response content
            output_text = response.choices[0].message.content

            # Attempt to extract token usage (if provided by DeepSeek)
            try:
                num_input_tokens = response.usage.prompt_tokens
                num_output_tokens = response.usage.completion_tokens
            except AttributeError:
                # If no usage info is in the response
                num_input_tokens = None
                num_output_tokens = None

            if count_token_usage:
                return output_text, num_input_tokens, num_output_tokens
            else:
                return output_text, None, None

        except Exception as e:
            logging.error(f"Error calling DeepSeek API: {e}")
            return "", None, None

    return query_model


def get_openai_query(
    model_name: str,
    openai_api_key: str,
    config: Dict,
    count_token_usage: bool = True,
    temperature=None,
) -> Callable:
    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=openai_api_key)

    def query_model(
        messages: List[Dict[str, str]],
        count_token_usage=count_token_usage,
        temperature=temperature,
    ) -> Tuple[str, int, int]:

        # Determine temperature value
        temp_value = (
            temperature if temperature is not None else config.get("temperature", 0.01)
        )

        try:
            # Create a completion request to OpenAI chat API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=config.get("max_tokens", 150),
                temperature=temp_value,
                n=1,  # Generate one completion per request
            )

            # Extract the assistant's response content
            output_text = response.choices[0].message.content

            # Extract token counts from the response if available
            num_input_tokens = response.usage.prompt_tokens
            num_output_tokens = response.usage.completion_tokens

            # Return the assistant's response and token counts
            if count_token_usage:
                return output_text, num_input_tokens, num_output_tokens
            else:
                return output_text, None, None

        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return "", None, None

    return query_model


def get_huggingface_query(
    model_id: str, config: Dict, count_token_usage: bool = False
) -> Callable:
    """
    Creates a query function for a HuggingFace model using the API.

    Args:
        model_id (str): The ID of the HuggingFace model to use.
        config (Dict): A dictionary of configuration parameters.

    Returns:
        Callable: A function that queries the model with a given message.
    """
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    ENDPOINT = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def query(
        payload,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
        rate_limit_delay: float = 2.0,
    ):
        retries = 0
        response = None

        while retries < max_retries:
            try:
                response = requests.post(ENDPOINT, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if response and response.status_code == 429:  # too many requests
                    logging.warning(
                        f"Rate limit hit. Sleeping for {rate_limit_delay} seconds."
                    )
                    time.sleep(rate_limit_delay)
                    rate_limit_delay *= 2  # Exponential backoff for rate limits
                else:
                    logging.warning(
                        f"HTTPError: {e}. Retrying in {backoff_factor * (2 ** retries)} seconds..."
                    )
                    time.sleep(
                        backoff_factor * (2**retries)
                    )  # Exponential backoff for other errors
                retries += 1
            except requests.exceptions.RequestException as e:
                logging.warning(
                    f"RequestException: {e}. Retrying in {backoff_factor * (2 ** retries)} seconds..."
                )
                time.sleep(backoff_factor * (2**retries))
                retries += 1

        logging.warning("Failed to get a valid response after retries.")
        return None

    def query_model(
        messages: List[Dict[str, str]],
        count_token_usage: bool = count_token_usage,
        temperature=None,
    ) -> Tuple[str, int, int]:

        # Determine temperature value
        temp_value = (
            temperature if temperature is not None else config.get("temperature", 0.01)
        )

        payload = {
            "inputs": messages,
            "parameters": {
                "do_sample": config.get("do_sample", False),
                "max_new_tokens": config.get("max_tokens", 100),
                "temperature": temp_value,
                "return_full_text": config.get("return_full_text", False),
            },
        }

        # Count input tokens if required
        num_input_tokens = (
            count_tokens(messages, tokenizer) if count_token_usage else None
        )

        response = query(payload)

        # Raise an exception if the query failed (i.e., response is None)
        if response is None:
            raise RuntimeError("Failed to get a valid response after retries.")

        output_text = response[0]["generated_text"]

        # Count output tokens if required
        num_output_tokens = (
            count_tokens(output_text, tokenizer) if count_token_usage else None
        )

        # Return based on token usage flag
        if count_token_usage:
            return output_text, num_input_tokens, num_output_tokens
        else:
            return output_text, None, None

    return query_model


def get_local_huggingface_query(
    model_name: str, config: Dict, count_token_usage: bool = False
) -> Callable:
    """
    Creates a local query function for a HuggingFace model.

    Args:
        model_name (str): The name of the HuggingFace model to use, loaded locally.
        config (Dict): A dictionary of configuration parameters for local execution.

    Returns:
        Callable: A function that locally queries the model with a given message.
    """
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Set dtype for mixed precision
    dtype = torch.float16 if config.get("use_mixed_precision", False) else torch.float32

    # Default parameters for loading the model
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": None,  # Ensure no offloading
        "trust_remote_code": True,
    }

    # Dynamically update model_kwargs with relevant config parameters
    valid_model_kwargs_keys = {"use_quantization", "use_flash_attention"}

    for key in valid_model_kwargs_keys:
        if key in config:
            if key == "use_quantization" and config[key]:
                try:
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                except ImportError:
                    logging.info(
                        "Quantization is not supported. Please install the necessary libraries or disable this feature."
                    )
            elif key == "use_flash_attention" and config[key]:
                model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)  # Move the model to the GPU after loading

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up the text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )

    def query_model(
        messages, count_token_usage: bool = count_token_usage, temperature=0.2
    ) -> Tuple[str, int, int]:
        generation_args = {
            "max_new_tokens": config.get("max_tokens", 500),
            "return_full_text": config.get("return_full_text", False),
            "temperature": temperature,
            "do_sample": config.get("do_sample", True),
        }

        # Concatenate messages into a single input string
        if isinstance(messages, str):
            input_text = messages
        elif isinstance(messages, list) and all(
            isinstance(msg, dict) and "text" in msg for msg in messages
        ):
            input_text = " ".join([msg["text"] for msg in messages])
        else:
            logging.error(
                "Invalid input type for 'messages'. Expected a string or a list of dictionaries with 'text' keys."
            )
            return "", None, None

        # Count input tokens if required
        num_input_tokens = (
            count_tokens(input_text, tokenizer) if count_token_usage else None
        )

        try:
            output = pipe(input_text, **generation_args)
            output_text = output[0]["generated_text"]
        except Exception as e:
            logging.error(f"Error during text generation: {e}")
            return "", None, None

        # Count output tokens if required
        num_output_tokens = (
            count_tokens(output_text, tokenizer) if count_token_usage else None
        )

        # Return based on token usage flag
        if count_token_usage:
            return output_text, num_input_tokens, num_output_tokens
        else:
            return output_text, None, None

    return query_model


def get_ollama_query(
    model_name: str,
    config: dict = None,
    count_token_usage: bool = False,
    temperature: float = None,
) -> Callable:
    """
    Creates a query function for an Ollama model.

    Args:
        model_name (str): The name of the Ollama model to use.
        config (Dict): A dictionary of configuration parameters.
        count_token_usage (bool): Whether to count the token usage.

    Returns:
        Callable: A function that queries the model with a given message.
    """

    def query_model(
        prompt: str,
        count_token_usage: bool = count_token_usage,
        temperature: float = temperature,
    ) -> Tuple[str, int, int]:
        # TODO: Implement token counting for Ollama
        if temperature is not None:
            response = ollama.generate(
                model=model_name, prompt=prompt, options={"temperature": temperature}
            )
        else:
            response = ollama.generate(model=model_name, prompt=prompt)
        result = response["response"]

        return result, None, None

    return query_model


def get_gemini_query(
    model_name: str,
    gemini_api_key: str,
    config: Dict,
    count_token_usage: bool = True,
    temperature=None,
) -> Callable:
    """
    Creates a query function for Google's Gemini API with retry mechanism.
    """
    import random
    import time

    from google import genai

    # Initialize the Gemini client with the API key
    client = genai.Client(api_key=gemini_api_key)

    def query_model(
        messages: List[Dict[str, str]],
        count_token_usage: bool = count_token_usage,
        temperature=temperature,
    ) -> Tuple[str, int, int]:

        # Determine temperature value
        temp_value = (
            temperature if temperature is not None else config.get("temperature", 0.01)
        )

        # Convert messages to a string if it's a list
        if isinstance(messages, list):
            # For OpenAI chat format, extract content
            if len(messages) == 1 and "content" in messages[0]:
                content = messages[0]["content"]
            else:
                # Fallback to combining all messages
                content = "\n".join([msg.get("content", "") for msg in messages])
        else:
            # For simple string format
            content = messages

        # Initialize retry parameters
        max_retries = 5
        base_delay = 2  # Start with 2 seconds
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Create a completion request to Gemini API
                response = client.models.generate_content(
                    model=model_name,
                    contents=content,
                )

                # Extract the output text
                output_text = response.text

                # Gemini doesn't directly provide token counts
                num_input_tokens = None
                num_output_tokens = None

                if count_token_usage:
                    return output_text, num_input_tokens, num_output_tokens
                else:
                    return output_text, None, None

            except Exception as e:
                error_message = str(e)
                logging.error(f"Error calling Gemini API: {e}")

                # If rate limit error, implement exponential backoff
                if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(
                            f"Maximum retries ({max_retries}) reached. Giving up."
                        )
                        break

                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                    logging.info(
                        f"Rate limit hit. Retrying in {delay:.2f} seconds (attempt {retry_count}/{max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    # For other errors, don't retry
                    break

        # If we get here, all retries failed or a non-retryable error occurred
        logging.error("Failed to get valid response from Gemini API after retries")
        return "", None, None

    return query_model


def get_query(model_name: str, model_type: str) -> Callable:
    """Get the correct model query function based on the model name and type.
    Params:
        model_name (str): The name of the model to use.
        model_type (str): The type of query to use. Either "openai", "huggingface",
                          "local_huggingface", "ollama", "deepseek", or "gemini".
    """
    if model_type == "ollama":
        return get_ollama_query(model_name, {})

    config = load_config(model_name)

    dotenv.load_dotenv()

    if model_type == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("No OpenAI API key found. Please set it in the .env file.")
        return get_openai_query(model_name, openai_api_key, config)
    elif model_type == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("No Gemini API key found. Please set it in the .env file.")
        return get_gemini_query(model_name, gemini_api_key, config)
    elif model_type == "local_huggingface":
        return get_local_huggingface_query(model_name, config)
    elif model_type == "huggingface":
        return get_huggingface_query(model_name, config)
    elif model_type == "deepseek":
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError(
                "No DeepSeek API key found. Please set it in the .env file."
            )
        return get_deepseek_query(model_name, deepseek_api_key, config)
    else:
        raise ValueError(
            f"Unknown query type: {model_type}. Please use 'openai', 'huggingface', 'local_huggingface', 'ollama', 'deepseek', or 'gemini'."
        )
