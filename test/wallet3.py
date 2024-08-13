from langchain.chains.openai_functions import create_structured_output_chain, create_openai_fn_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json
import requests

def call_endpoint(api_url, endpoint, request_data):
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(f'{api_url}/{endpoint}', headers=headers, json=request_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print(f'{api_url}/{endpoint}', request_data)
        return None

functions = [
    {
        "name": "swap",
        "description": "Swaps tokens between two tokens on the same chain.",
        "parameters": {
            "type": "object",
            "properties": {
                "chainName": {
                    "type": "string",
                    "description": "The ID of the blockchain chain where the token swap will occur.",
                },
                "sourceAmount": {
                    "type": "string",
                    "description": "The amount of tokens to be swapped from the sourceToken.",
                },
                "sourceToken": {
                    "type": "string",
                    "description": "The token to be swapped from.",
                },
                "destinationToken": {
                    "type": "string",
                    "description": "The token to be received in the swap.",
                },
            },
            "required": ["chainName", "sourceAmount", "sourceToken", "destinationToken"],
        },
    },
    {
        "name": "bridge",
        "description": "Bridges tokens between different blockchain networks.",
        "parameters": {
            "type": "object",
            "properties": {
                "sourceChainName": {
                    "type": "string",
                    "description": "The ID of the source blockchain chain.",
                },
                "destinationChainName": {
                    "type": "string",
                    "description": "The ID of the destination blockchain chain.",
                },
                "sourceToken": {
                    "type": "string",
                    "description": "The token to be transferred from the source chain to the destination chain.",
                },
                "destinationToken": {
                    "type": "string",
                    "description": "The token to be received on the destination chain.",
                },
                "sourceAmount": {
                    "type": "string",
                    "description": "The amount of tokens to be transferred.",
                },
            },
            "required": ["sourceChainName", "destinationChainName", "sourceToken", "destinationToken", "sourceAmount"],
        },
    },
    {
        "name": "transfer",
        "description": "Transfers tokens to another address on the same chain.",
        "parameters": {
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "description": "The token to be transferred.",
                },
                "amount": {
                    "type": "string",
                    "description": "The amount of tokens to be transferred.",
                },
                "recipient": {
                    "type": "string",
                    "description": "The recipient's address for the token transfer.",
                },
                "chainName": {
                    "type": "string",
                    "description": "The ID of the blockchain chain where the token transfer will occur.",
                },
            },
            "required": ["token", "amount", "recipient", "chainName"],
        },
    }
]

llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, openai_api_key='')
prompt_msgs = [
    SystemMessage(
        content="You are a world class algorithm for extracting information in structured formats."
    ),
    HumanMessage(content="Use the given format to extract information from the following input:"),
    HumanMessagePromptTemplate.from_template("{input}"),
    HumanMessage(content="Tips: Make sure to answer in the correct format"),
]
prompt = ChatPromptTemplate(messages=prompt_msgs)
chain = create_openai_fn_chain(functions, llm, prompt, verbose=False)
messages = ["Swap 1 ETH for USDC on Ethereum", "Bridge 1 ETH from Ethereum to Optimism", "Transfer 1 ETH to 0xC5a05570Da594f8edCc9BEaA2385c69411c28CBe", "Swap 1 ETH for ARB on Ethereum then bridge to Arbitrum"]
for message in messages[3:]:
    data = chain.run(message)
    print(data)
    response_data = call_endpoint("https://wallet.spicefi.xyz/v1", data['name'], data['arguments'])
    print(response_data)