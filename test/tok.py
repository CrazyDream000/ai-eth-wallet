import tiktoken
import json

with open('test/functions9.json', 'r') as f:
    docs = json.load(f)

def ntfs(string):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
    num_tokens = len(encoding.encode(string))
    return num_tokens

x = ntfs(json.dumps(docs))

behavior = """
    Solve the user's intent by executing a series of steps.
    Ensure steps are scheduled and triggered when necessary.
    Ensure all parts of the user's intent are solved.
    "all" is a valid input amount.
    "half" is a valid input amount.
    Only use the functions you have been provided with.
    Only use the function inputs you have been provided with.
    Respond with a single sentence.
    Every function call should be unique.
    """
y = ntfs(behavior)

# print(x, y, x + (1.2 * y))

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    with open('test/functions9.json', 'r') as f:
        docs = json.load(f)
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
            if key == "function_call":
                num_tokens += len(encoding.encode(json.dumps(docs)))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

z = "I have bought JONES tokens with half of your ETH tokens, deposited them into the ETH-JONES pool on the Sushi protocol, and then traded the LP tokens for plsJones tokens."
z1 = json.dumps({'inputAmount': 'half', 'inputToken': 'ETH', 'outputToken': 'JONES'})
z2 = json.dumps({'protocolName': 'Sushi', 'poolName': 'ETH-JONES', 'amount': 'outputAmount', 'token': 'JONES'})
z3 = json.dumps({'inputAmount': 'outputAmount', 'inputToken': 'outputToken', 'outputToken': 'plsJones'})
print(ntfs(z)+ntfs(z1)+ntfs(z2)+ntfs(z3))