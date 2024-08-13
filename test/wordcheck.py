import json
from collections import Counter

with open('test/newwords', 'r') as f:
    words = f.read().split("\n")
keywords = [word.strip().lower() for word in words if word.strip()]
keywords.extend(["transfer", "swap", "bridge", "deposit", "withdraw", "claim", "borrow", 
"lend", "repay", "stake", "unstake", "long", "short", "lock", "unlock", "vote", "buy", "sell"])
keywords.extend(["condition", "time", "token", "protocol", "chain", "pool", "apy", "ltv", "loan", "outputtokens", "outputamount", "outputtoken"])
protocols = {"uniswap", "sushiswap", "pancakeswap", "kyberswap",
                "gmx", "aave", "kwenta", "compound", "radiant", "morpho", "rocket pool", 
                "jonesdao", "rodeo", "kyber", "mmf", "camelot", "curve", 
                "thena", "plutus", "trader joe", "solidly", 
                "redacted cartel", "blur", "balancer", "yearn", "chronos", 
                "dolomite", "lodestar", "stargate", "pendle", "sushi", "velodrome", 
                "gmd", "dodo finance", "paraspace", "hop", "hop protocol", "redacted", "spice",
                "polylend", "wing-finance", "paraspace-lending-v1", "woofi", "symbiosis finance",
                "mav", "spice finance", "nitro cartel", "rodeo finance", "defillama",
                "lido", "timeswap", "gnd", "tigristrade", "dodoex", "bungee", "syncswap",
                "llamaswap", "deepp", "aura", "coinbase", "binance", "jones dao", "jones"} # global protocols
chains = {"mainnet", "ethereum", "base", "arbitrum", "optimism", "avalanche", "zksync", 
            "bnb", "bsc", "binance", "polygon", "gnosis", "fantom", "canto", "mantle", "linea", "opbnb",
            "starknet", "cosmos"} # global chains
shortened_chains = {"arb": "arbitrum", "eth": "ethereum"}
pools = {"eth-usdc", "tricrypto", "weth-grail", "eth-jones", "pt-glp", "usdc-usdt", "wjaura"}
tokens = ["eth", "usdc", "usdt", "dai", "weth", "btc", "wbtc", "op", "ohm", "grail", 
            "glp", "uni", "pepe", "pls", "stg", "arb", "jones", "plsjones", "lp", 
            "pt", "smp", "lmp", "xgrail", "bitcoin", "btrfly",
            "dpx", "rdpx", "plsdpx", "coin", "sweed", "dodo", "saint", "vethe",
            "rlbtrfly", "livethe", "tigusd", "usdc.e", "usdce", "bal", "spnft"] # global tokens
with open('test/coins.json', 'r') as f:
    q = json.load(f)
    tokens.extend([x['symbol'].lower() for x in q if x['symbol'] not in keywords and x['symbol'] != '' and not any(z in x['symbol'] for z in ['.', '+', '*', '?', '^', '$', '(', ')', '[', ']', '{', '}', '|', '\\']) and not all(zz.isdigit() for zz in x['symbol'])])
    # tokens.extend([x['name'].lower() for x in q if x['name'] not in keywords and x['name'] != '' and not any(z in x['name'] for z in ['.', '+', '*', '?', '^', '$', '(', ')', '[', ']', '{', '}', '|', '\\']) and not all(zz.isdigit() for zz in x['name'])])    
with open('test/newanswers.json', 'r') as f:
    data = [y for x in list(json.load(f).keys()) for y in x.replace(',','').replace('.','').replace('%','').replace(' pm ',' ').replace(' am ',' ').replace('pm ',' ').replace('am ',' ').split(' ')]
counts = dict(Counter(data))
for x in counts:
    if x not in keywords and x not in tokens and x not in protocols and x not in chains and x not in pools and not x.isdigit():
        print(x, counts[x])