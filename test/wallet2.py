import requests
import json

def get_call(message):
    api_key = ''
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    with open('functions.json', 'r') as file:
        functions = json.load(file)
    data = {
        "model": "gpt-3.5-turbo-0613",
        "messages": [
            {"role": "user", "content": f"{message}"}
        ],
        "functions": functions
    }

    # Make the API call
    response = requests.post(url, headers=headers, json=data)

    # Check the response status code
    if response.status_code == 200:
        result = response.json()
        # print(result)
        return result['choices'][0]['message']['function_call']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

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

if __name__ == "__main__":
    api_url = "https://wallet.spicefi.xyz/v1"
    messages = ["Swap 1 ETH for USDC on Ethereum", "Bridge 1 ETH from Ethereum to Optimism", "Transfer 1 ETH to 0xC5a05570Da594f8edCc9BEaA2385c69411c28CBe", "Swap 1 ETH for ARB on Ethereum then bridge to Arbitrum"]
    for message in messages[3:]:
        data = get_call(message)
        print(data)
        response_data = call_endpoint(api_url, data['name'], json.loads(data['arguments']))
        print(response_data)
