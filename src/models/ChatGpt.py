import configparser

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential


# Function to count the number of tokens in a list of messages
def num_tokens_from_messages(messages):
    encoding = tiktoken.get_encoding("cl100k_base")

    if isinstance(messages, str):
        return len(encoding.encode(messages))

    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Each message has a fixed token cost
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens -= 1  # Reduce 1 token if there's a name (role is required)
    num_tokens += 1  # Add 1 token for the reply delimiter
    return num_tokens


# Function to remove a specified number of tokens from the end of text
def remove_tokens_from_end(text, num_tokens):
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    if num_tokens >= len(tokens):
        return ""
    else:
        tokens = tokens[:-num_tokens]
        return encoding.decode(tokens)


# Function to call GPT model in streaming mode with retries
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def call_gpt_stream(conversation, model="gpt-4", temperature=0, presence_penalty=0, condition=None):
    chat_response = openai.ChatCompletion.create(
        model=model,
        messages=conversation,
        temperature=temperature,
        presence_penalty=presence_penalty,
        stream=True,
    )

    result = ""
    for chunk in chat_response:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            result += delta["content"]
        if condition is not None and condition in result:
            return None

    return result


# Function to call GPT model with retries and improved error handling
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def call_gpt(conversation, model="gpt-4", temperature=0, presence_penalty=0):
    response = openai.ChatCompletion.create(
        temperature=temperature,
        presence_penalty=presence_penalty,
        model=model,
        messages=conversation,
    )

    result = {}
    if response["choices"][0].get("finish_reason", "") == "stop":
        answer = response["choices"][0]["message"]["content"]
        if "```" in answer:
            try:
                result = response.loads(answer.split('```')[1])
            except Exception as e:
                print("Error parsing JSON:", e)
        else:
            try:
                result = response.loads(answer)
            except Exception as e:
                print("Error parsing JSON:", e)
    else:
        print("RESULT HAS NOT BEEN COMPLETED")
        print(response)

    return result


# Read configuration from config files
config = configparser.ConfigParser()
config.read(['configs/config.cfg', 'configs/config.dev.cfg'])

openai_settings = config['OPENAI']

# Set OpenAI organization ID and API key from config
openai.organization = openai_settings["OPENAI_ORGANIZATION_ID"]
openai.api_key = openai_settings["OPENAI_API_KEY"]
