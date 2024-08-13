import requests
import json
from niylib import w3
from hexbytes import HexBytes as hb
import os
from dotenv import load_dotenv

load_dotenv()


def fetch_contract_sc(contract, api_key):
    url = "https://api.etherscan.io/api"
    parameters = {
        "module": "contract",
        "action": "getsourcecode",
        "address": contract,
        "apikey": api_key
    }
    try:
        response = requests.get(url, params=parameters)
        response_json = response.json()
        return response_json
    except requests.exceptions.RequestException as e:
        print("Error occurred during API call:", e)
        return None

def get_contract_data(contract):
    api_key = os.getenv("ETHERSCAN_API_KEY")
    response_data = fetch_contract_abi(contract, api_key)
    if response_data['status'] != '1':
        print(response_data)
        return []
    else:
        abi = json.loads(response_data['result'])
        output_data = []
        # print(abi)
        for entry in abi:
            if entry['type'] == 'function':
                func_name = entry['name']
                func_description = f"Description for {func_name}"
                parameters = entry['inputs']

                transformed_parameters = []
                for param in parameters:
                    param_name = param['name']
                    param_type = param['internalType']
                    param_description = f"Description for {param_name}"

                    transformed_parameters.append({
                        "name": param_name,
                        "type": "string",  # You can adjust the type based on the actual data type
                        "description": param_description
                    })

                transformed_entry = {
                    "name": func_name,
                    "description": func_description,
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }

                for param in transformed_parameters:
                    transformed_entry["parameters"]["properties"][param["name"]] = {
                        "type": param["type"],
                        "description": param["description"]
                    }

                transformed_entry["parameters"]["required"] = [param["name"] for param in transformed_parameters]
                output_data.append(transformed_entry)

        return output_data


x = fetch_contract_sc("0xF1Cd4193bbc1aD4a23E833170f49d60f3D35a621", os.getenv("ETHERSCAN_API_KEY"))
with open('test/dump.json', 'w') as f:
    json.dump(x, f)
print(json.loads(x['result'][0]['SourceCode'][1:-1])['sources'].keys())
for k in list(json.loads(x['result'][0]['SourceCode'][1:-1])['sources'].keys()):
    os.makedirs(os.path.dirname(f'test/aave/{k}'), exist_ok=True)
    with open(f'test/aave/{k}', 'w') as f:
        s = json.loads(x['result'][0]['SourceCode'][1:-1])['sources'][k]['content']
        print(type(s))
        s = s.replace('0.8.10', '>=0.8.10')
        f.write(s)
# with open('Pool.sol', 'w') as f:
    # f.write(json.loads(x['result'][0]['SourceCode'][1:-1])['sources']['lib/aave-v3-core/contracts/protocol/pool/Pool.sol']['content'])
# with open('PoolStorage.sol', 'w') as f:
    # f.write(json.loads(x['result'][0]['SourceCode'][1:-1])['sources']['lib/aave-v3-core/contracts/protocol/pool/PoolStorage.sol']['content'])
