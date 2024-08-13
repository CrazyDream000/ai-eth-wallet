import requests
import copy
import json

# flag - vote TVL doesn't count
# flag - @niyant you noted that token/outputToken will always be provided. unsure if true given difficulty involved

# global fee dictionary
fees = {
    "transfer": 0,
    "swap": 0.005,
    "bridge": 0.005,
    "deposit": 0,
    "withdraw": 0,
    "claim": 0.005,
    "borrow": 0.005,
    "lend": 0.005,
    "repay": 0.005,
    "stake": 0.005,
    "unstake": 0.005,
    "long": 0.005,
    "short": 0.005,
    "lock": 0.005,
    "unlock": 0.005,
    "vote": 0,
    "claim": 0.005,
    "loop": 0.035,
    "condition": 0.01,
    "time": 0.01,
    2: 0.005,
    3: 0.0075,
    4: 0.01,
    5: 0.0125,
}

def clean_transactions(raw_transactions):
    """
    Calculates the fee incurred by user.

    :param raw_transactions: list, transactions where each transaction is a dictionary 
    :return: list, transactions where each transaction is a dictionary
    """

    transactions = copy.deepcopy(raw_transactions)
    for idx, t in enumerate(transactions):

        # classifies by type
        if transactions[idx]["name"] in ("transfer", "bridge", "deposit", 
                                         "withdraw", "borrow", "lend", 
                                         "repay", "stake", "unstake",
                                         "lock", "unlock", "claim"):
            transactions[idx]["type"] = "regular"
        elif transactions[idx]["name"] in ("swap", "long", "short", "loop"):
            transactions[idx]["type"] = "trade"
        elif transactions[idx]["name"] in ("condition","time"):
            transactions[idx]["type"] = "conditional"
        else:
            transactions[idx]["type"] = "vote"

        # standardizes chain
        if transactions[idx]["name"] == "bridge":
            transactions[idx]["chain1"] = transactions[idx]["args"]["sourceChainName"]
        elif transactions[idx]["type"] in ("regular", "trade", "vote"):
            transactions[idx]["chain1"] = transactions[idx]["args"]["chainName"]
        else:
            transactions[idx]["chain1"] = None
        
        # standardized token
        if transactions[idx]["type"] == "regular": 
            transactions[idx]["token1"] = raw_transactions[idx]["args"]["token"]
            transactions[idx]["amount1"] = raw_transactions[idx]["args"]["amount"]
        elif transactions[idx]["type"] == "trade": 
            transactions[idx]["token1"] = raw_transactions[idx]["args"]["inputToken"]
            transactions[idx]["amount1"] = raw_transactions[idx]["args"]["inputAmount"]
        else:
            transactions[idx]["token1"] = None

        # runs dependency check
        transactions[idx]["dependent"] = is_dependent(transactions, idx)

        # gets price
        if transactions[idx]["type"] in ("regular", "trade"):
            try:
                if transactions[idx]["args"]["contractAddress"] == '0x0000000000000000000000000000000000000000':
                    transactions[idx]["price"] = get_native_token_price(get_native_token(transactions[idx]["chain1"]))
                else:
                    transactions[idx]["price"] = get_contract_address_price(get_coin_gecko_chain(transactions[idx]["chain1"]), transactions[idx]["args"]["contractAddress"])
            except:
                transactions[idx]["price"] = 0
        else:
            transactions[idx]["price"] = 0

        # replaces outputAmount, turns all amounts into floats
        if transactions[idx]["amount1"] == "outputAmount":
            if transactions[idx]["name"] == "swap":
                transactions[idx]["amount1"] = transactions[idx - 1] * transactions[idx]["price"] / transactions[idx - 1]["price"]
            elif transactions[idx]["name"] == "bridge":
                transactions[idx]["amount1"] = transactions[idx-1]["amount1"]
            else:
                transactions[idx]["amount1"] = 0
        transactions[idx]["amount1"] = float(transactions[idx]["amount1"])

    print(transactions)
    
    return transactions


def is_dependent(transactions, idx):
    """
    Checks whether the current transaction is dependent on the prior transaction. Defaults to dependency unless conditions are met for independency.

    :param transactions: list, transactions where each transaction is a dictionary
    :param idx: int, id of the current transactions being tested
    :return: boolean, True if is dependent, False if is independent
    """

    if idx > 0:
        if transactions[idx - 1]["name"] == "transfer":
            return False
        elif transactions[idx - 1]["type"] == "trade":
            if transactions[idx]["token1"] != transactions[idx - 1]["args"]["outputToken"]:
                return False
    else:
        return False
    return True


def get_native_token(chainName):
    """
    Gets the name of the native token related to a chain from functions.json

    :param chainName: str, name of the starting chain from functions.json
    :return: str, native token name compatible with CoinGecko API
    """

    native_tokens = {
        "ethereum": "ethereum",
        "mainnet": "ethereum",
        "bsc": "bnb",
        "bnb": "bnb",
        "arbitrum": "ethereum",
        "polygon": "ethereum",
        "optimism": "ethereum",
        "avalanche": "avax",
        "base": "ethereum",
        "gnosis": "ethereum",
        "zksync": "ethereum",
        "mantle": "ethereum",
        "linea": "ethereum",
        "metis": "ethereum",
        "polygon zkevm": "ethereum",
        "canto": "canto",
        "opbnb": "bnb",
        "fantom": "FTM"
    }
    
    try:
        return native_tokens[chainName]
    except:
        print("The chain cannot be matched to a coingecko token.")
    
    return None

def get_coin_gecko_chain(chainName):
    """
    Gets the name of the coin gecko matched chain (asset_platform) related to a chain from functions.json

    :param chainName: str, name of the chain from functions.json
    :return: str, asset_platform/chain id compatible with CoinGecko API
    """
    
    coin_gecko_chains = {
        "ethereum": "ethereum",
        "mainnet": "ethereum",
        "bsc": "binance-smart-chain",
        "bnb": "binance-smart-chain",
        "arbitrum": "arbitrum-one",
        "polygon": "polygon-pos",
        "optimism": "optimistic-ethereum",
        "avalanche": "avalanche",
        "base": "base",
        "gnosis": "xdai",
        "zksync": "zksync",
        "mantle": "mantle",
        "linea": "linea",
        "metis": "metis-andromeda",
        "polygon zkevm": "polygon-zkevm",
        "canto": "canto",
        "opbnb": "opbnb",
        "fantom": "fantom"
    }

    try:
        return coin_gecko_chains[chainName]
    except:
        print("The chain cannot be matched to a coingecko chain.")

    return None


def get_contract_address_price(asset_platform, contract_address, currency='usd'):
    """
    Get the current price of a cryptocurrency.
    
    :param asset_platform: str, the id of the platform issuing tokens (see asset_platforms endpoint for list of options)
    :param contract_address: str, the contract address of the token
    :param currency: str, the fiat currency to convert to (default is 'usd')
    :return: float, the current price in the specified fiat currency
    """
    
    url = f"https://api.coingecko.com/api/v3/simple/token_price/{asset_platform}?contract_address={contract_address}&vs_currencies={currency}"

    response = requests.get(url)
    data = response.json()
    
    if contract_address in data:
        return data[contract_address][currency]
    else:
        raise ValueError("Cannot get price for contract address: {contract_address}")
    

def get_native_token_price(token_id, currency='usd'):
    """
    Get the current price of a cryptocurrency.
    
    :param token_id: str, the id of the native_token
    :param currency: str, the fiat currency to convert to (default is 'usd')
    :return: float, the current price in the specified fiat currency
    """
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies={currency}"

    response = requests.get(url)
    data = response.json()
    
    if token_id in data:
        return data[token_id][currency]
    else:
        raise ValueError("Cannot get price for token id: {token_id}")


def calc_tvl(transactions):
    """
    Gets the total tvls of the sub-sets of the transaction set

    :param transactions: list, transactions where each transaction is a dictionary
    :return: float, tvl of transaction sets in USD terms
    """
    
    # determine fee take points
    for idx, t in enumerate(transactions):
        
        # calculate tvl of a string of transactions
        tvl = 0 
        typ = transactions[idx]["type"]
        price = transactions[idx]["price"] 
        
        if typ in ("trade", "regular"):     
            if transactions[idx]["dependent"]:
                tvl = max(tvl, transactions[idx]["amount1"] * price)
            else:
                tvl += transactions[idx]["amount1"] * price
    
    return tvl


def run_fee_calculations(raw_transactions):
    """
    Calculates the fee incurred by user.

    :param transactions: list, transactions where each transaction is a dictionary
    :return: float, fee amount to charge user excluding gas in native token based on the chain where fees are taken
    :return: string, chain from relevant transactions determining where to charge the user
    """
 
    transactions = clean_transactions(raw_transactions)
    min_tvl = 1000 # determined to be USD based
    discount_fee = 0
    tvl = calc_tvl(transactions)

    for idx,  t in enumerate(transactions):
        if t["type"] == "conditional":
            continue
        else:
            starting_chain = transactions[idx]["chain1"] # to know which chain we charge the fee on
            break
    
    # apply growth pipeline discount 
    if tvl < min_tvl and starting_chain == "ethereum":
        return discount_fee, starting_chain

    # adds a combination multiplier based on the number of 'actual' transactions in the transaction combo
    condition_counter = 0
    time_counter = 0
    action_counter = 0
    action_fees_usd = 0
    
    for idx, t in enumerate(transactions):
        if t["type"] == "condition":
            condition_counter += 1
        elif t["type"] == "time":
            time_counter += 1
        else:            
            action_fees_usd += fees[t["name"]] * t["price"] * t["amount1"]
            action_counter += 1
        
    global_additive_fee_percentage = 0
    if action_counter > 5:
        global_additive_fee_percentage += fees[5]
    elif action_counter < 2:
        pass
    else:
        global_additive_fee_percentage += fees[action_counter]
    global_additive_fee_percentage += (fees["condition"] * condition_counter) + (fees["time"] * time_counter) 

    # converts the fee_percentage into an absolute fee to be applied on the native token
    native_token = get_native_token(starting_chain)
    if native_token == None:
        absolute_fee_native_token = 0
    else:
        try:
            absolute_fee_native_token = ((global_additive_fee_percentage * tvl) + (action_fees_usd)) / get_native_token_price(native_token)
        except:
            absolute_fee_native_token = 0

    return absolute_fee_native_token, starting_chain



def get_names_from_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    names = []
    for key in data:
        try:
            name = data[key][0][0]["name"]
            if name not in names:
                names.append(name)
        except:
            continue
    return names

print(get_names_from_json('/Users/jakubjaniak/walletai/test/newanswers.json'))