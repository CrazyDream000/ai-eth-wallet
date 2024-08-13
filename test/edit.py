import json
import sys

def strip(data):
    """
    Remove optional placeholder words
    """
    if isinstance(data, str):
        if data in ['inputToken', 'outputToken', 'inputAmount', 'outputAmount', 'protocolName', 'chainName', 'sourceChainName', 'destinationChainName', 'poolName', 'leverageMultiplier', 'inputAmountUnits', 'numberLoops']:
            return data
        return data.lower()
    elif isinstance(data, list):
        return [strip(item) for item in data]
    elif isinstance(data, dict):
        return {key: strip(value) for key, value in data.items()}
    else:
        return data
    
    
def standardize(data):
    """
    Use to standardize data to match ground truth answers if necessary
    """
    if isinstance(data, str):
        if data in ['inputToken', 'outputToken', 'inputAmount', 'outputAmount', 'protocolName', 'chainName', 'sourceChainName', 'destinationChainName', 'poolName', 'leverageMultiplier', 'inputAmountUnits', 'numberLoops']:
            return data
        return data.lower()
    elif isinstance(data, list):
        return [standardize(item) for item in data]
    elif isinstance(data, dict):
        return {standardize(key): standardize(value) for key, value in data.items()}
    else:
        return data

if len(sys.argv) != 3:
    print("Usage: python script.py 70 71")
    sys.exit(1)

try:
    inputs = [int(arg) for arg in sys.argv[1:3]]
except ValueError:
    print("Invalid input. Please provide two integer values.")
    sys.exit(1)

with open(f"test/res{inputs[0]}.json", "r") as f:
    data = json.load(f)
with open(f"test/res{inputs[1]}.json", "w") as f:
    json.dump(strip(standardize(data)), f)
    
with open("test/newanswers.json", "r") as f:
    data = json.load(f)
with open("test/answers2.json", "w") as f:
    json.dump(strip(standardize(data)), f)
