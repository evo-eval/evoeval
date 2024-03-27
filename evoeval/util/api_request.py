import signal
import time
from typing import Dict, Union

import openai
import tiktoken
from google.generativeai import GenerationConfig
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

client = openai.OpenAI()


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config):
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = client.chat.completions.create(**config)
            signal.alarm(0)
        except openai._exceptions.BadRequestError as e:
            print(e)
            signal.alarm(0)
        except openai._exceptions.RateLimitError as e:
            print("Rate limit exceeded. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(5)
        except openai._exceptions.APIConnectionError as e:
            print("API connection error. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(1)
    return ret


def create_gemini_config(
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
) -> Dict:
    config = GenerationConfig(
        candidate_count=batch_size,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    return config


safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def request_gemini_engine(model, message, config):
    ret = None
    count = 0
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = model.generate_content(
                message, generation_config=config, safety_settings=safety_settings
            )
            s = ret.text  # check if response can be accessed.
            signal.alarm(0)
        except Exception as e:
            ret = None  # reset
            print("Unknown error. Waiting...")
            count += 1
            print(e)
            # here we need to slightly increase temperature to combat weird gemini output of
            # The token generation was stopped as the response was flagged for unauthorized citations.
            if count > 10:
                config.temperature = min(config.temperature + 0.1, 1)
            signal.alarm(0)
            time.sleep(20)
    return ret


def create_palm_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    model: str = "models/text-bison-001",
) -> Dict:
    config = {
        "model": model,
        "prompt": message,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "safety_settings": [
            {
                "category": HarmCategory.HARM_CATEGORY_DEROGATORY,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_TOXICITY,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUAL,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_VIOLENCE,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_MEDICAL,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
        ],
    }
    return config


def request_palm_engine(model, config):
    ret = None
    count = 0
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = model.generate_text(**config)
            s = ret.result  # check if response can be accessed.
            if s is None:
                config["temperature"] = min(config["temperature"] + 0.1, 1)
                count += 1
                if count > 100:
                    ret.result = ""  # just return empty string
                else:
                    ret = None  # reset
            signal.alarm(0)
        except Exception as e:
            ret = None  # reset
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(20)
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    model: str = "claude-2.1",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": message}],
        }
    return config


def request_anthropic_engine(client, config):
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = client.messages.create(**config)
            signal.alarm(0)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(10)
    return ret
