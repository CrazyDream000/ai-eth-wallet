import requests
import json
from niylib import w3
from hexbytes import HexBytes as hb
import os
from dotenv import load_dotenv

load_dotenv()


def fetch_contract_abi(contract, api_key):
    url = "https://api.etherscan.io/api"
    parameters = {
        "module": "contract",
        "action": "getabi",
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


print(get_contract_data("0xF1Cd4193bbc1aD4a23E833170f49d60f3D35a621"))
