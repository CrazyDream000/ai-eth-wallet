import json
import os
import time
import openai
from dotenv import load_dotenv
import re
import sys
import zoneinfo
from datetime import datetime
import psycopg
import requests

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def transfer(**kwargs):
    return "/transfer has been called successfully"
def swap(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def bridge(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def deposit(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def withdraw(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def claim(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def borrow(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def lend(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def repay(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def stake(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def unstake(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def long(**kwargs):
    return "/long has been called successfully"
def short(**kwargs):
    return "/short has been called successfully"
def close(**kwargs):
    return "/close has been called successfully"
def lock(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})
def unlock(**kwargs):
    return json.dumps({"outputAmount": "outputAmount"})
def vote(**kwargs):
    return "/vote has been called successfully"
def condition(**kwargs):
    return "/condition has been called successfully"
def tim(**kwargs):
    return "/time has been called successfully"

def timezones():
    summer = datetime.fromisoformat('2023-06-21')
    winter = datetime.fromisoformat('2023-12-21')
    ans = set()
    for tz in zoneinfo.available_timezones():
        tz0 = zoneinfo.ZoneInfo(tz).tzname(summer)
        tz1 = zoneinfo.ZoneInfo(tz).tzname(winter)
        if tz0[0] not in ['+', '-'] and '/' not in tz0:
            ans.add(tz0.lower())
        if tz1[0] not in ['+', '-'] and '/' not in tz1:
            ans.add(tz1.lower())
    return list(ans)

def to_lowercase(data):
    if isinstance(data, str):
        if data == 'outputAmount' or data == 'outputToken':
            return data
        return data.lower()
    elif isinstance(data, list):
        return [to_lowercase(item) for item in data]
    elif isinstance(data, dict):
        return {key: to_lowercase(value) for key, value in data.items()}
    else:
        return data

def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))
    if lw == 0:
        lw = 1e9
    # return a tuple
    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    try:
        return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]
    except Exception as e:
        print(v1, v2)
        raise e

def check_cache(message):
    db_config = {
        'dbname': os.getenv('DATABASE_NAME'),
        'user': os.getenv('DATABASE_USER'),
        'password': os.getenv('DATABASE_PASSWORD'),
        'host': os.getenv('DATABASE_HOST'),
        'port': int(os.getenv('DATABASE_PORT', 5432))
    }
    with psycopg.connect(**db_config) as conn:
        res = conn.execute("SELECT id, generated_api_calls, edited_api_calls from tracking where inputted_query=%s;", (message, )).fetchall()
        for r in res:
            found = conn.execute("SELECT generated_correct, edited_correct from dataset where query_id=%s and (generated_correct = True or edited_correct = True) order by updated desc;", (r[0], )).fetchone()
            if found[0]:
                return r[1]
            if found[1]:
                return r[2]
    return []

def track_query(address, message, test=True):
    db_config = {
        'dbname': os.getenv('DATABASE_NAME'),
        'user': os.getenv('DATABASE_USER'),
        'password': os.getenv('DATABASE_PASSWORD'),
        'host': os.getenv('DATABASE_HOST'),
        'port': int(os.getenv('DATABASE_PORT', 5432))
    }
    with psycopg.connect(**db_config) as conn:
        if not test:
            r = conn.execute("INSERT INTO tracking (user_address, inputted_query, created, updated) values (%s, %s, %s, %s) returning id;", (address, message, int(time.time()), int(time.time()))).fetchone()[0]
            conn.execute("INSERT INTO dataset (query_id, created, updated) values (%s, %s, %s);", (r, int(time.time()), int(time.time())))
        else:
            r = -1
            print(address, message)
    return r

def track_response(r, answer, test=True):
    db_config = {
        'dbname': os.getenv('DATABASE_NAME'),
        'user': os.getenv('DATABASE_USER'),
        'password': os.getenv('DATABASE_PASSWORD'),
        'host': os.getenv('DATABASE_HOST'),
        'port': int(os.getenv('DATABASE_PORT', 5432))
    }
    with psycopg.connect(**db_config) as conn:
        if not test:
            conn.execute("UPDATE tracking set generated_api_calls = %s where id = %s;", (json.dumps(answer), r))
        else:
            print(answer)

def make_request_with_retries(url, max_retries=3, initial_delay=1):
    for retry in range(max_retries + 1):
        try:
            response = requests.get(url)
            
            if response.status_code == 200 and response.json().get("status") == "success":
                data = response.json()  # Parse the response as JSON
                return data  # Return the parsed JSON response
            else:
                print(f"Request failed with status code {response.status_code}")
                print("Response content:")
                print(response.text)  # Print the response content
        except Exception as e:
            print(f"Request failed with exception: {e}")
        
        if retry < max_retries:
            # Calculate the exponential backoff delay
            delay = initial_delay * (2 ** retry)
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print("Max retries reached. Request failed.")
    
    return None  # Return None if all retries fail


def run_conversation(message, prompt, save=0):
    results = []
    behavior = """
    Solve the user's intent by executing a series of steps.
    Ensure steps are scheduled and triggered when necessary.
    Ensure all parts of the user's intent are solved.
    "all" is a valid input amount.
    Only use the functions you have been provided with.
    Only use the function inputs you have been provided with.
    Respond with a single sentence.
    Every function call should be unique.
    """
    behavior = [
        {"role": "system", "content": behavior},
    ]
    query = [
        {"role": "user", "content": message}
    ]
    messages = behavior + query
    with open('test/functions8-min.json', 'r') as file:
        functions = json.load(file)
    tokens = 0
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        messages=messages,
        functions=functions,
        function_call="auto",
        timeout=10
    )
    tokens += response["usage"]["prompt_tokens"]
    # print(tokens)
    response_message = response["choices"][0]["message"]
    while response_message.get("function_call") and not response_message.get("content"):
        print(response_message)
        if len(messages) > 1 and response_message == messages[-2]:
            print('REPEAT')
            print(response_message, messages[-2])
            break
        if len(messages) > 3 and response_message == messages[-4] and messages[-2] == messages[-6]:
            print('REPEAT')
            print(response_message, messages[-2])
            print(messages[-4], messages[-6])
            break
        results.append(response_message['function_call'])
        available_functions = {
            "transfer": transfer,
            "swap": swap,
            "bridge": bridge,
            "deposit": deposit,
            "withdraw": withdraw,
            "claim": claim,
            "borrow": borrow,
            "lend": lend,
            "repay": repay,
            "stake": stake,
            "unstake": unstake,
            "long": long,
            "short": short,
            "close": close,
            "lock": lock,
            "unlock": unlock,
            "vote": vote,
            "condition": condition,
            "time": tim
        }
        if response_message["function_call"]["name"] == "buy":
            response_message["function_call"]["name"] = "swap"
        if response_message["function_call"]["name"] == "sell":
            response_message["function_call"]["name"] = "swap"
        function_name = response_message["function_call"]["name"]
        if function_name not in available_functions:
            break
        function_to_call = available_functions[function_name]
        response_message["function_call"]["arguments"] = re.sub(r'\[outputAmount\]', r'["outputAmount"]', response_message["function_call"]["arguments"])
        try:
            function_args = json.loads(response_message["function_call"]["arguments"])
        except Exception as e:
            print(response_message)
            raise e
        if "slippage" not in message and "slippage" in function_args:
            del function_args['slippage']
            response_message["function_call"]["arguments"] = json.dumps(function_args)
        function_response = function_to_call(**function_args)
        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0,
            messages=messages,
            functions=functions,
            function_call="auto",
            timeout=10
        )
        tokens += response["usage"]["prompt_tokens"]
        # print(tokens)
        response_message = response["choices"][0]["message"]
        # if tokens > 4097:
            # print(f'{tokens} TOKENS DETECTED')
        if len(messages) >= 20:
            break
    messages.append(response_message)
    return results, response_message

def perform_message(message, prompt=1, save=0):
    # che = check_cache(message)
    # if che != []:
        # for c in che:
            # print(c['name'], c['args'])
        # return "", "" 
    if message == "":
        return "", ""
    user_address = ""
    # message_id = track_query(user_address, message, test=True)
    om = message
    message = message.lower()
    nl = []
    ml = message.split(" ")
    for l in ml:
        if l[-1] == "$":
            nl.append("$" + l[:-1])
        else:
            nl.append(l)
    message = " ".join(nl)
    message = message.replace("'", "")
    message = message.replace(" the ", " ")
    message = message.replace(" that ", " ")
    message = message.replace("compound my ", "claim and redeposit ")
    message = message.replace(" my ", " ")
    message = message.replace(" me ", " ")
    rxwd = r'(?P<wkdy>monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
    message = re.sub(r'\s+(\d+)\s+(january|february|march|april|may|june|july|august|september|october|november|december)', r' time \2 \1', message)
    message = re.sub(r'(each|every)\s*' + rxwd, r'time \g<wkdy> recurrence weekly', message)
    message = re.sub(r'(each|every|once a)\s*' + r'week', r'time recurrence weekly', message)
    message = re.sub(r'(each|every|once a)\s*' + r'day', r'time recurrence daily', message)
    # message = message.replace("or, if ", ", condition operator or ")
    # message = message.replace("or,if ", ", condition operator or ")
    # message = message.replace("or if ", ", condition operator or ")
    message = message.replace(", if ", ", condition ")
    message = message.replace(",if ", ", condition ")
    message = message.replace(" if ", ", condition ")
    message = re.sub("^if ", "condition ", message)
    # message = message.replace("or, when ", ", condition operator or ")
    # message = message.replace("or,when ", ", condition operator or ")
    # message = message.replace("or when ", ", condition operator or ")
    message = message.replace(", when ", ", condition ")
    message = message.replace(",when ", ", condition ")
    message = message.replace(" when ", ", condition ")
    message = re.sub("^when ", "condition ", message)
    message = message.replace(", once ", ", condition ")
    message = message.replace(",once ", ", condition ")
    message = message.replace(" once ", ", condition ")
    message = re.sub("^once ", "condition ", message)
    message = message.replace(", as soon as ", ", condition ")
    message = message.replace(",as soon as ", ", condition ")
    message = message.replace(" as soon as ", ", condition ")
    message = re.sub("^as soon as ", "condition ", message)
    # message = message.replace(", at", ", time")
    # message = message.replace(",at", ", time")
    # message = message.replace(" at", ", time")
    message = re.sub("^at ", "time ", message)
    message = message.replace(" ape", " buy")
    message = re.sub("^ape ", "buy ", message)
    message = message.replace(" convert", " swap")
    message = re.sub("^convert ", "swap ", message)
    # message = message.replace(" sell", " swap")
    # message = re.sub("^sell ", "swap ", message)
    message = message.replace(" dump", " sell")
    message = re.sub("^dump ", "sell ", message)
    message = message.replace(" grab", " claim")
    message = re.sub("^grab ", "claim ", message)
    message = message.replace(" harvest", " claim")
    message = re.sub("^harvest ", "claim ", message)
    message = message.replace(" move", "transfer")
    message = re.sub("^move ", "transfer ", message)
    message = message.replace("relock", "lock")
    message = message.replace("revest", "lock")
    message = message.replace("unvest", "unlock")
    message = message.replace("restake", "stake")
    message = message.replace("remove", "withdraw")
    message = message.replace(" half", " 'half' ")
    message = message.replace(" rewards ", " rewards token ")
    message = message.replace("compound it ", "claim and redeposit it ")
    message = message.replace("compound the ", "claim and redeposit the ")
    message = message.replace("compound rewards ", "claim and redeposit rewards ")
    message = message.replace("compound into ", "claim and redeposit into ")
    message = message.replace("claim and claim", "claim")
    message = message.replace(" via ", " using ")
    message = message.replace(" lock positions", " positions")
    message = message.replace(" mcap ", " market cap ")
    message = message.replace(" ether ", " eth ")
    message = message.replace(" farm ", " deposit ")
    unders = [' below ', ' is under ', ' lt ', ' lte ', ' less than ', ' less_than ', ' lessThan ', ' lessthan ', ' goes below ', ' sub ', ' is at or below ']
    overs = [' above ', ' is over ', ' gt ', ' gte ', ' greater than ', ' greater_than ', ' greaterThan ', ' greaterthan ', ' goes above ', ' is at or above ']
    equals = [' equal ', ' hits ', ' hit ', ' is at ', ' reaches ']
    notequals = [' notequals ']
    for under in unders:
        message = message.replace(under, " <= ")
    for over in overs:
        message = message.replace(over, " >= ")
    for equal in equals:
        message = message.replace(equal, " == ")
    for notequal in notequals:
        message = message.replace(notequal, " != ")
    message = message.replace("is <=", "<=")
    message = message.replace("is >=", ">=")
    message = message.replace("is ==", "==")
    message = message.replace("is !=", "!=")
    message = message.replace(",,", ",")
    rxnm = r'(?P<num>\d+\.\d*|\d+|\.\d+)'
    rxnm2 = r'(?P<num2>\d+\.\d*|\d+|\.\d+)'
    message = re.sub(rxnm + ',' + rxnm2, r'\g<num>\g<num2>', message)
    # message = re.sub(r'\$(\d+\.\d*|\d+|\.\d+)', r'price \1', message)
    message = re.sub(r'(is|at)\s*\$(\d+\.\d*|\d+|\.\d+)', r'== \2 usd', message)
    message = re.sub(r'\$(\d+\.\d*|\d+|\.\d+)', r'\1 usd', message)
    message = re.sub(r'(\d+\.\d*|\d+|\.\d+)\%', r"'\1%'", message)
    message = message.replace("$", "")
    message = message.replace("''", "'")
    message = message.replace("  ", " ")
    message = message.replace("  ", " ")
    message = message.replace("eth mainnet", "ethereum")
    message = message.replace("ethereum mainnet", "ethereum")
    message = message.replace("mainnet", "ethereum")
    message = message.replace("zk sync", "zksync")
    actions = ["transfer", "swap", "bridge", "deposit", "withdraw", "claim", "borrow", 
    "lend", "repay", "stake", "unstake", "long", "short", "lock", "unlock", "vote", "buy", "sell", "close"]
    rxact = r'(?P<act>' + r'|'.join(actions) + r')'
    message = re.sub(r'\.\s*' + rxact, r'. then \g<act>', message)
    keywords = ["condition", "time", "token", "protocol", "chain", "pool", "apy", "ltv", "loan", 
    "outputtokens", "outputamount", "outputtoken", "redeposit", "amount_units"]
    keywords.extend(actions)
    with open("test/newwords", "r") as file:
        words = file.read().split("\n")
    keywords.extend([word.strip().lower() for word in words if word.strip()])
    keywords.extend([f"{word.strip().lower()}s" for word in words if word.strip()])
    keywords.extend([f"{word.strip().lower()}es" for word in words if word.strip()])
    keywords.extend([f"{word.strip().lower()}d" for word in words if word.strip()])
    keywords.extend([f"{word.strip().lower()}ed" for word in words if word.strip()])
    keywords.extend([f"{word.strip().lower()}ing" for word in words if word.strip()])
    keywords.extend(timezones())
    keywords.extend(["ust"])
    keywords = set(keywords)
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
                 "llamaswap", "deepp", "aura", "coinbase", "binance", "jones dao", "jones",
                 "rocketpool"} # global protocols
    chains = {"mainnet", "ethereum", "base", "arbitrum", "optimism", "avalanche", "zksync", 
              "bnb", "bsc", "binance", "polygon", "gnosis", "fantom", "canto", "mantle", "linea", "opbnb",
              "starknet", "cosmos"} # global chains
    shortened_chains = {"arb": "arbitrum", "eth": "ethereum"}
    pools = {"eth-usdc", "tricrypto", "weth-grail", "eth-jones", "pt-glp", "usdc-usdt", "wjaura"}
    tokens = ["eth", "usdc", "usdt", "dai", "weth", "btc", "wbtc", "op", "ohm", "grail", 
              "glp", "uni", "pepe", "pls", "stg", "arb", "jones", "plsjones", "lp", 
              "pt", "smp", "lmp", "xgrail", "bitcoin", "btrfly",
              "dpx", "rdpx", "plsdpx", "coin", "sweed", "dodo", "saint", "vethe",
              "rlbtrfly", "livethe", "tigusd", "usdc.e", "usdce", "bal", "spnft", "wig",
              "gwei", "merit circle", "fxs"] # global tokens
    with open('test/coins.json', 'r') as f:
        q = json.load(f)
        tokens.extend([x['symbol'].lower() for x in q if x['symbol'] not in keywords and x['symbol'] != '' and not any(z in x['symbol'] for z in ['.', '+', '*', '?', '^', '$', '(', ')', '[', ']', '{', '}', '|', '\\']) and not all(zz.isdigit() for zz in x['symbol'])])
        # tokens.extend([x['name'].lower() for x in q if x['name'] not in keywords and x['name'] != '' and not any(z in x['name'] for z in ['.', '+', '*', '?', '^', '$', '(', ')', '[', ']', '{', '}', '|', '\\']) and not all(zz.isdigit() for zz in x['name'])])
    ve = make_request_with_retries("https://wallet.spicefi.xyz/v1/verified-entities")
    if ve:
        for a in ve['actions']:
            keywords.add(a.lower())
        for p in ve['protocols']:
            if 'name' in p:
                protocols.add(p['name'].lower())
            if 'pools' in p:
                for k, v in list(p['pools'].items()):
                    for kv in list(v.keys()):
                        pools.add(kv.lower())
        for c in ve['chains']:
            if 'name' in c:
                chains.add(c['name'].lower())
            if 'tokens' in c:
                for t in c['tokens']:
                    if t['symbol'] not in keywords and t['symbol'] != '' and not any(z in t['symbol'] for z in ['.', '+', '*', '?', '^', '$', '(', ')', '[', ']', '{', '}', '|', '\\']) and not all(zz.isdigit() for zz in t['symbol']):
                        tokens.append(t['symbol'].lower())
    assert "mog" not in keywords
    assert "mog" != ''
    assert not any(z in "mog" for z in ['.', '+', '*', '?', '^', '$', '(', ')', '[', ']', '{', '}', '|', '\\'])
    assert not all(zz.isdigit() for zz in "mog")
    assert "mog" in tokens, tokens
    orderedtokens = tokens
    tokens = set(tokens)
    assert "aura" in protocols
    assert "aura" in tokens
    confusing1 = protocols & tokens
    confusing2 = chains & tokens
    # rxnm = r'(?P<num>\d+)'
    rxtk = r'(?P<tkn>' + r'|'.join(tokens) + r')'
    rxtk2 = r'(?P<tkn2>' + r'|'.join(tokens) + r')'
    message = re.sub(rxnm + r'e\b', r'\g<num>eth', message)
    message = re.sub(rxnm + rxtk + ' ', r'\g<num> \g<tkn> ', message)
    message = re.sub(rxtk + ' - ' + rxtk2 + ' ', r'\g<tkn>-\g<tkn2> ', message)
    rxdr = r'(?P<dur>min(ute)?s?|s(ec)?(ond)?s?|h(ou)?rs?|days?|w(ee)?ks?)'
    message = re.sub(rxnm + rxdr, r'\g<num> \g<dur>', message)
    message = re.sub(rxnm + r'\s*k\b', r'\g<num>000', message)
    message = re.sub(rxnm + r'\s*(m|mn)\b', r'\g<num>000000', message)
    message = re.sub(rxnm + r'\s*(b|bn)\b', r'\g<num>000000000', message)
    message = re.sub(r'\s+(lp)\s+(\d+\.\d*|\d+|\.\d+)\b', r' deposit \2', message)
    message = re.sub(r'^lp\s+(\d+\.\d*|\d+|\.\d+)\b', r'deposit \1', message)
    message = re.sub(r'\s+(lp)\s+' + rxtk + r'\b', r' deposit \g<tkn>', message)
    message = re.sub(r'^lp\s+' + rxtk + r'\b', r'deposit \g<tkn>', message)
    message = re.sub(r'(,|\.)\s*(lp)\s+into\b', r'\1 deposit outputtokens into', message)
    message = re.sub(r'(\s+|-)' + rxtk + r'\s*lp\b', r'\1\g<tkn>_lp token', message)
    message = re.sub(r'\s*usd\s+' + rxtk + r'\b', r' amount_units usd of \g<tkn>', message)
    message = re.sub(r'\s+(\d+\.\d*|\d+|\.\d+)\s*market\s*cap\b', r' \1_market_cap', message)
    rxar = r'(?P<addr>0x[\da-fA-F]{40}|\w+\.eth\b)'
    rxm = r'\b(?![0-9]+x\b)\w+\b'
    premsglist = re.findall(rxm, re.sub(rxar, r'', message).replace(',','').replace('.','').replace('%','').replace(' pm ',' ').replace(' am ',' ').replace('pm ',' ').replace('am ',' '))
    msglist = [[x, word2vec(x)] for x in premsglist if x not in keywords and x not in tokens and x not in protocols and x not in chains and x not in pools and not x.isdigit()]
    print(msglist)
    oo = {ot: word2vec(ot) for ot in orderedtokens}
    # ta = {t: word2vec(t) for t in tokens}
    pb = {p: word2vec(p) for p in protocols}
    cc = {c: word2vec(c) for c in chains}
    od = {o: word2vec(o) for o in pools}
    ae = {a: word2vec(a) for a in actions}
    cd = {**pb, **cc, **od, **ae, **oo}
    updates = {}
    for m in msglist:
        mcd = 0
        grp = 0
        for k, v in list(cd.items()):
            csd = cosdis(m[1], v)
            if csd > max(mcd + (0.04*grp), 0.8):
                if k in pb:
                    grp = 4
                if k in cc:
                    grp = 3
                if k in od:
                    grp = 2
                if k in ae:
                    grp = 1
                updates[m[0]] = k
                mcd = csd
    updated = {"tokens": [], "protocols": [], "chains": [], "pools": []}
    for u, v in list(updates.items()):
        if v in tokens:
            updated['tokens'].append(v)
        if v in protocols:
            updated['protocols'].append(v)
        if v in chains:
            updated['chains'].append(v)
        if v in pools:
            updated['pools'].append(v)
        message = message.replace(f" {u} ", f" {v} ")
        message = message.replace(f" {u}.", f" {v}.")
        message = message.replace(f" {u},", f" {v},")
        message = re.sub(f" {u}$", f" {v}", message)
        message = re.sub(f"^{u} ", f"{v} ", message)
    for t in tokens:
        if t in confusing1 or t in confusing2:
            continue
        message = message.replace(f" {t} ", f" {t} token ")
        message = message.replace(f" {t}. ", f" {t} token. ")
        message = message.replace(f" {t}, ", f" {t} token, ")
        et = re.escape(t)
        try:
            message = re.sub(f" {et}$", f" {t} token", message)
        except Exception as e:
            print(t)
            raise e
        message = re.sub(rf" {et}\.$", f" {t} token.", message)
        message = re.sub(f"^{et} ", f"{t} token ", message)
        message = re.sub(f"{et} price", f"{t}_price", message)
        message = re.sub(f"price of {et}", f"price_of_{t}", message)
        message = re.sub(f"{et} balance ", f"{t}_balance ", message)
        message = re.sub(f"balance of {et}", f"balance_of_{t}", message)
        message = re.sub(f"{et} market cap", f"{t}_market_cap", message)
        message = re.sub(f"market cap of {et}", f"market_cap_of_{t}", message)
    for p in protocols:
        message = re.sub(r'(\d+\.\d*|\d+|\.\d+|all|half)\s*' + f'{p}', f'{p} token', message)
        message = message.replace(f" {p} ", f" {p} protocol ")
        message = message.replace(f" {p}.", f" {p} protocol.")
        message = message.replace(f" {p},", f" {p} protocol,")
        message = re.sub(f" {p}$", f" {p} protocol", message)
        message = re.sub(f"^{p} ", f"{p} protocol ", message)
    for c in chains:
        message = message.replace(f" {c} ", f" {c} chain ")
        message = message.replace(f" {c}.", f" {c} chain.")
        message = message.replace(f" {c},", f" {c} chain,")
        message = re.sub(f" {c}$", f" {c} chain", message)
        message = re.sub(f"^{c} ", f"{c} chain ", message)
    for o in pools:
        message = re.sub(r'protocol\s*' + f'{o} token', f'{o} pool', message)
        message = message.replace(f" {o} ", f" {o} pool ")
        message = message.replace(f" {o}.", f" {p} pool.")
        message = message.replace(f" {o},", f" {o} pool,")
        message = re.sub(f" {o}$", f" {o} pool", message)
        message = re.sub(f"^{o} ", f"{o} pool ", message)
    message = message.replace("protocol finance", "finance")
    message = message.replace("protocol token", "token")
    message = message.replace("protocol cartel", "cartel")
    message = message.replace("protocol rewards", "rewards")
    message = message.replace("protocol dao", "dao protocol")
    message = message.replace("protocol exchange", "protocol")
    message = message.replace("protocol protocol", "protocol")
    message = message.replace("token finance", "finance")
    message = message.replace("token staking", "staking")
    message = message.replace("token position", "position")
    message = message.replace("token cartel", "cartel")
    message = message.replace("token protocol", "protocol")
    message = message.replace("token chain", "chain")
    message = message.replace("token pool", "pool")
    message = message.replace("token rewards token", "token")
    message = message.replace("token token", "token")
    message = message.replace("pool pool", "pool")
    message = message.replace("pool lp", "lp")
    message = message.replace("chain network", "chain")
    message = message.replace("chain chain", "chain")
    print(om)
    print(message)
    # return "", ""
    try:
        results, response = run_conversation(message, prompt, save)
    except Exception as e:
        print(e)
        time.sleep(10)
        results, response = run_conversation(message, prompt, save)
    # print('Human Response:')
    if results == []:
        print(response['content'])
    # print('Transactions to Sign:')
    signs = []
    for data in results:
        name = data['name']
        args = to_lowercase(json.loads(data['arguments']))
        if name == 'time' and args == {"start_time": "now"}:
            continue
        if name == 'swap' and 'amount' in args and 'inputAmount' not in args:
            args['inputAmount'] = args['amount']
            del args['amount']
        if name not in ['condition', 'time'] and 'operator' in args and ('value' not in args or "start_time" not in args or "recurrence" not in args):
            del args['operator']
        if name == 'bridge' and 'chainName' in args and 'sourceChainName' not in args:
            args['sourceChainName'] = args['chainName']
            del args['chainName']
        if 'amount_units' in args and 'token' in args and args['amount_units'].lower() == args['token'].lower():
            del args['amount_units']
        if 'leverageMultiplier' in args and args['leverageMultiplier'].lower() not in message.lower():
            del args['leverageMultiplier']
        if 'amount' in args and type(args['amount']) == str and args['amount'].lower() == 'rewardsamount':
            args['amount'] = 'outputAmount'
        if 'token' in args and type(args['token']) == str and args['token'].lower() == 'rewardstoken':
            args['amount'] = 'outputToken'
        if 'inputAmount' in args and type(args['inputAmount']) == str and args['inputAmount'].lower() == 'rewardsamount':
            args['amount'] = 'outputAmount'
        if 'inputToken' in args and type(args['inputToken']) == str and args['inputToken'].lower() == 'rewardstoken':
            args['amount'] = 'outputToken'
        if 'recurrence' in args and 'type' in args['recurrence']:
            args['recurrence']['type'] = args['recurrence']['type'].replace('hours', 'hourly')
        if 'amount_units' in args and args['amount_units'].lower() not in message.lower():
            del args['amount_units']
        if 'token' in args:
            if type(args['token']) == list:
                randomlist = True
                for c, z in enumerate(args['token']):
                    if z[-1] != c+1:
                        randomlist = False
                if randomlist:
                    args['token'] = 'all'
            if type(args['token']) == str and args['token'] != 'outputToken' and args['token'] != 'outputtoken':
                args['token'] = args['token'].lower().replace(" token", "")
                args['token'] = args['token'].lower().replace("token ", "")
                args['token'] = args['token'].lower().replace("_token", "")
                args['token'] = args['token'].lower().replace("token", "")
                args['token'] = args['token'].lower().replace(" protocol", "")
                args['token'] = args['token'].lower().replace("_protocol", "")
                args['token'] = args['token'].lower().replace("protocol", "")
                args['token'] = args['token'].lower().replace(" coin", "")
                args['token'] = args['token'].lower().replace("_lp", " lp")
                if " lp" in args['token']:
                    updated['tokens'].append(args['token'])
                args['token'] = re.sub("^ ", "", args['token'].lower())
                args['token'] = re.sub(" $", "", args['token'].lower())
            elif type(args['token']) == list:
                new = []
                for tok in args['token']:
                    if tok == 'outputToken' or tok == 'outputtoken':
                        new.append(tok)
                        continue
                    tok = tok.lower().replace(" token", "")
                    tok = tok.lower().replace("token ", "")
                    tok = tok.lower().replace("_token", "")
                    tok = tok.lower().replace("token", "")
                    tok = tok.lower().replace(" protocol", "")
                    tok = tok.lower().replace("_protocol", "")
                    tok = tok.lower().replace("protocol", "")
                    tok = tok.lower().replace(" coin", "")
                    tok = tok.lower().replace("_lp", " lp")
                    if " lp" in tok:
                        updated['tokens'].append(tok)
                    tok = re.sub("^ ", "", tok.lower())
                    tok = re.sub(" $", "", tok.lower())
                    new.append(tok)
                if len(new) > 1:
                    args['token'] = new
                elif len(new) == 1:
                    args['token'] = new[0]
                else:
                    print(1)
                    del args['token']
        if 'inputToken' in args:
            if type(args['inputToken']) == list:
                randomlist = True
                for c, z in enumerate(args['inputToken']):
                    if z[-1] != str(c+1):
                        randomlist = False
                if randomlist:
                    args['inputToken'] = 'all'
            if type(args['inputToken']) == str and args['inputToken'] != 'outputToken' and args['inputToken'] != 'outputtoken':
                args['inputToken'] = args['inputToken'].lower().replace(" token", "")
                args['inputToken'] = args['inputToken'].lower().replace("token ", "")
                args['inputToken'] = args['inputToken'].lower().replace("_token", "")
                args['inputToken'] = args['inputToken'].lower().replace("token", "")
                args['inputToken'] = args['inputToken'].lower().replace(" protocol", "")
                args['inputToken'] = args['inputToken'].lower().replace("_protocol", "")
                args['inputToken'] = args['inputToken'].lower().replace("protocol", "")
                args['inputToken'] = args['inputToken'].lower().replace(" coin", "")
                args['inputToken'] = args['inputToken'].lower().replace("_lp", " lp")
                if " lp" in args['inputToken']:
                    updated['tokens'].append(args['inputToken'])
                args['inputToken'] = re.sub("^ ", "", args['inputToken'].lower())
                args['inputToken'] = re.sub(" $", "", args['inputToken'].lower())
            elif type(args['inputToken']) == list:
                new = []
                for tok in args['inputToken']:
                    if tok == 'outputToken' or tok == 'outputtoken':
                        new.append(tok)
                        continue
                    tok = tok.lower().replace(" token", "")
                    tok = tok.lower().replace("token ", "")
                    tok = tok.lower().replace("_token", "")
                    tok = tok.lower().replace("token", "")
                    tok = tok.lower().replace(" protocol", "")
                    tok = tok.lower().replace("_protocol", "")
                    tok = tok.lower().replace("protocol", "")
                    tok = tok.lower().replace(" coin", "")
                    tok = tok.lower().replace("_lp", " lp")
                    if " lp" in tok:
                        updated['tokens'].append(tok)
                    tok = re.sub("^ ", "", tok.lower())
                    tok = re.sub(" $", "", tok.lower())
                    new.append(tok)
                if len(new) > 1:
                    args['inputToken'] = new
                elif len(new) == 1:
                    args['inputToken'] = new[0]
                else:
                    del args['inputToken']
        if 'outputToken' in args and ('token' in args['outputToken'].lower() or 'protocol' in args['outputToken'].lower() or 'coin' in args['outputToken'].lower()):
            args['outputToken'] = args['outputToken'].lower().replace(" token", "")
            args['outputToken'] = args['outputToken'].lower().replace("token ", "")
            args['outputToken'] = args['outputToken'].lower().replace("_token", "")
            args['outputToken'] = args['outputToken'].lower().replace("token", "")
            args['outputToken'] = args['outputToken'].lower().replace(" protocol", "")
            args['outputToken'] = args['outputToken'].lower().replace("_protocol", "")
            args['outputToken'] = args['outputToken'].lower().replace("protocol", "")
            args['outputToken'] = args['outputToken'].lower().replace(" coin", "")
            args['outputToken'] = args['outputToken'].lower().replace("_lp", " lp")
            if " lp" in args['outputToken']:
                updated['tokens'].append(args['outputToken'])
            args['outputToken'] = re.sub("^ ", "", args['outputToken'].lower())
            args['outputToken'] = re.sub(" $", "", args['outputToken'].lower())
        if 'targetToken' in args and ('token' in args['targetToken'].lower() or 'protocol' in args['targetToken'].lower() or 'coin' in args['targetToken'].lower()):
            args['targetToken'] = args['targetToken'].lower().replace(" token", "")
            args['targetToken'] = args['targetToken'].lower().replace("token ", "")
            args['targetToken'] = args['targetToken'].lower().replace("_token", "")
            args['targetToken'] = args['targetToken'].lower().replace("token", "")
            args['targetToken'] = args['targetToken'].lower().replace(" protocol", "")
            args['targetToken'] = args['targetToken'].lower().replace("_protocol", "")
            args['targetToken'] = args['targetToken'].lower().replace("protocol", "")
            args['targetToken'] = args['targetToken'].lower().replace(" coin", "")
            args['targetToken'] = args['targetToken'].lower().replace("_lp", " lp")
            if " lp" in args['targetToken']:
                updated['tokens'].append(args['targetToken'])
            args['targetToken'] = re.sub("^ ", "", args['targetToken'].lower())
            args['targetToken'] = re.sub(" $", "", args['targetToken'].lower())
        if 'subject' in args and ('token' in args['subject'].lower() or 'protocol' in args['subject'].lower()):
            args['subject'] = args['subject'].lower().replace(" token", "")
            args['subject'] = args['subject'].lower().replace("token ", "")
            args['subject'] = args['subject'].lower().replace("_token", "")
            args['subject'] = args['subject'].lower().replace("token", "")
            args['subject'] = args['subject'].lower().replace(" protocol", "")
            args['subject'] = args['subject'].lower().replace("_protocol", "")
            args['subject'] = args['subject'].lower().replace("protocol", "")
            args['subject'] = args['subject'].lower().replace(" coin", "")
            args['subject'] = args['subject'].lower().replace("_lp", " lp")
        if 'value' in args and ('token' in args['value'].lower() or 'protocol' in args['value'].lower()):
            args['value'] = args['value'].lower().replace(" token", "")
            args['value'] = args['value'].lower().replace("token ", "")
            args['value'] = args['value'].lower().replace("_token", "")
            args['value'] = args['value'].lower().replace("token", "")
            args['value'] = args['value'].lower().replace(" protocol", "")
            args['value'] = args['value'].lower().replace("_protocol", "")
            args['value'] = args['value'].lower().replace("protocol", "")
            args['value'] = args['value'].lower().replace(" coin", "")
            args['value'] = args['value'].lower().replace("_lp", " lp")
        if 'chainName' in args and 'chain' in args['chainName'].lower():
            args['chainName'] = args['chainName'].lower().replace(" chain", "")
        if 'sourceChainName' in args and 'chain' in args['sourceChainName'].lower():
            args['sourceChainName'] = args['sourceChainName'].lower().replace(" chain", "")
        if 'destinationChainName' in args and 'chain' in args['destinationChainName'].lower():
            args['destinationChainName'] = args['destinationChainName'].lower().replace(" chain", "")
        if 'targetChainName' in args and 'chain' in args['targetChainName'].lower():
            args['targetChainName'] = args['targetChainName'].lower().replace(" chain", "")
        if 'protocolName' in args and 'protocol' in args['protocolName'].lower():
            args['protocolName'] = args['protocolName'].lower().replace(" protocol", "")
        if 'poolName' in args:
            args['poolName'] = args['poolName'].lower().replace(" pool", "")
            args['poolName'] = args['poolName'].lower().replace("_pool", "")
            args['poolName'] = args['poolName'].lower().replace("pool", "")
            args['poolName'] = args['poolName'].lower().replace(" token", "")
            args['poolName'] = args['poolName'].lower().replace("token ", "")
            args['poolName'] = args['poolName'].lower().replace("_token", "")
            args['poolName'] = args['poolName'].lower().replace("token", "")
            args['poolName'] = args['poolName'].lower().replace(" protocol", "")
            args['poolName'] = args['poolName'].lower().replace("_protocol", "")
            args['poolName'] = args['poolName'].lower().replace("protocol", "")
            args['poolName'] = args['poolName'].replace('_lp', '')
            args['poolName'] = args['poolName'].replace(' lp', '')
            if args['poolName'] == "":
                del args['poolName']
        if name != 'bridge' and 'protocolName' in args and 'chainName' not in args and args['protocolName'].lower() in chains:
            args['chainName'] = args['protocolName']
            args['protocolName'] = 'all'
        if name == 'bridge' and 'protocolName' in args and 'sourceChainName' not in args and args['protocolName'].lower() in chains:
            args['sourceChainName'] = args['protocolName']
            del args['protocolName']
        if 'token' in args and type(args['token']) == str:
            if args['token'] == 'rewards':
                args['token'] = 'outputToken'
            if args['token'] != '' and args['token'].lower() != 'outputToken'.lower() and args['token'] != 'all' and args['token'].lower() not in message.lower() and args['token'].lower() not in updated['tokens']:
                print(args['token'], updated['tokens'], message)
                print(2)
                del args['token']
        elif 'token' in args and type(args['token']) == list:
            new = []
            for it in args['token']:
                if it == 'rewards':
                    it = 'outputToken'
                if it != '' and it.lower() != 'outputToken'.lower() and it != 'all' and it.lower() not in message.lower() and it.lower() not in updated['tokens']:
                    continue
                if 'outputToken' in args and it == args['outputToken']:
                    continue
                new.append(it)
            if len(new) > 1:
                args['token'] = new
            elif len(new) == 1:
                args['token'] = new[0]
            else:
                print(3)
                del args['token']
        if 'inputToken' in args and type(args['inputToken']) == str:
            if args['inputToken'] == 'rewards':
                args['inputToken'] = 'outputToken'
            if args['inputToken'] != '' and args['inputToken'].lower() != 'outputToken'.lower() and args['inputToken'] != 'all' and args['inputToken'].lower() not in message.lower() and args['inputToken'].lower() not in updated['tokens']:
                del args['inputToken']
        elif 'inputToken' in args and type(args['inputToken']) == list:
            new = []
            for it in args['inputToken']:
                if it == 'rewards':
                    it = 'outputToken'
                if it != '' and it.lower() != 'outputToken'.lower() and it != 'all' and it.lower() not in message.lower() and it.lower() not in updated['tokens']:
                    # print(1)
                    continue
                if 'outputToken' in args and it == args['outputToken']:
                    # print(2)
                    continue
                new.append(it)
            if len(new) > 1:
                args['inputToken'] = new
            elif len(new) == 1:
                args['inputToken'] = new[0]
            else:
                del args['inputToken']
        if 'outputToken' in args and args['outputToken'].lower() not in message.lower() and args['outputToken'].lower() not in updated['tokens']:
            del args['outputToken']
        if 'targetToken' in args and args['targetToken'].lower() not in message.lower() and args['targetToken'].lower() not in updated['tokens']:
            del args['targetToken']
        if 'protocolName' in args and args['protocolName'] != '' and args['protocolName'] != 'all' and args['protocolName'].lower() not in message.lower() and args['protocolName'].lower() not in updated['protocols']:
            del args['protocolName']
        if 'protocolName' in args and 'position' in args['protocolName'].lower():
            args['protocolName'] = args['protocolName'].lower().replace('position', '')
            args['protocolName'] = args['protocolName'].lower().replace(' position', '')
        if 'protocolName' in args and args['protocolName'] != '' and args['protocolName'] != 'all' and args['protocolName'].lower() not in protocols:
            if name in ['claim', 'lock'] and args['protocolName'].lower() in tokens:
                args['token'] = args['protocolName'].lower()
            del args['protocolName']
        if 'poolName' in args and args['poolName'] != '' and args['poolName'] != 'all' and args['poolName'].lower() not in message.lower() and args['poolName'].lower() not in updated['pools']:
            del args['poolName']
        if 'poolName' in args and args['poolName'].replace(' ','') == 'rewards':
            del args['poolName']
        if 'poolName' in args and args['poolName'] != '' and args['poolName'] != 'all':
            if name in ['claim', 'lock'] and args['poolName'].lower() in tokens and 'token' not in args:
                args['token'] = args['poolName'].lower()
                del args['poolName']
        if 'chainName' in args and args['chainName'] != '' and args['chainName'].lower() not in message.lower() and args['chainName'].lower() not in updated['chains']:
            del args['chainName']
        if 'sourceChainName' in args and args['sourceChainName'] != '' and args['sourceChainName'].lower() not in message.lower() and args['sourceChainName'].lower() not in updated['chains']:
            del args['sourceChainName']
        if 'destinationChainName' in args and args['destinationChainName'] != '' and args['destinationChainName'].lower() not in message.lower() and args['destinationChainName'].lower() not in updated['chains']:
            del args['destinationChainName']
        if 'chainName' in args and args['chainName'] != '' and args['chainName'].lower() not in chains:
            if args['chainName'] in shortened_chains:
                args['chainName'] = shortened_chains[args['chainName']]
            else:
                del args['chainName']
        if 'sourceChainName' in args and args['sourceChainName'] != '' and args['sourceChainName'].lower() not in chains:
            if args['sourceChainName'] in shortened_chains:
                args['sourceChainName'] = shortened_chains[args['sourceChainName']]
            else:
                del args['sourceChainName']
        if 'destinationChainName' in args and args['destinationChainName'] != '' and args['destinationChainName'].lower() not in chains:
            if args['destinationChainName'] in shortened_chains:
                args['destinationChainName'] = shortened_chains[args['destinationChainName']]
            else:
                del args['destinationChainName']
        if 'inputAmount' in args and type(args['inputAmount']) == str:
            if args['inputAmount'] != '' and args['inputAmount'].lower() != 'outputAmount'.lower() and args['inputAmount'] != 'all' and args['inputAmount'] != 'half':
                if args['inputAmount'].lower() not in message.lower():
                    del args['inputAmount']
                else:
                    try:
                        y = float(args['inputAmount'].replace('%','').replace('$','').replace(',',''))
                    except:
                        yy = args['inputAmount'].split(' ')
                        if len(yy) > 1:
                            try:
                                y = float(yy[0].replace('%','').replace('$','').replace(',',''))
                                args['inputAmount'] = yy[0]
                            except:
                                del args['inputAmount']
                        else:
                            if args['inputAmount'] in tokens and (('inputToken' in args and args['inputToken'] == '') or 'inputToken' not in args):
                                args['inputToken'] = args['inputAmount']
                            del args['inputAmount']
        elif 'inputAmount' in args and type(args['inputAmount']) == list:
            new = []
            for ia in args['inputAmount']:
                if ia != '' and ia.lower() != 'outputAmount'.lower() and ia != 'all' and ia != 'half':
                    if ia.lower() not in message.lower():
                        continue
                    else:
                        try:
                            y = float(ia.replace('%','').replace('$','').replace(',',''))
                        except:
                            yy = ia.split(' ')
                            if len(yy) > 1:
                                try:
                                    y = float(yy[0].replace('%','').replace('$','').replace(',',''))
                                    ia = yy[0]
                                except:
                                    continue
                            else:
                                continue
                new.append(ia)
            if len(new) > 1:
                args['inputAmount'] = new
            elif len(new) == 1:
                args['inputAmount'] = new[0]
            else:
                del args['inputAmount']
        if 'amount' in args and type(args['amount']) == str:
            if args['amount'] != '' and args['amount'].lower() != 'outputAmount'.lower() and args['amount'] != 'all' and args['amount'] != 'half':
                if args['amount'].lower() not in message.lower():
                    del args['amount']
                else:
                    try:
                        y = float(args['amount'].replace('%','').replace('$','').replace(',',''))
                    except:
                        if args['amount'] in tokens and (('token' in args and args['token'] == '') or 'token' not in args):
                            args['token'] = args['amount']
                        del args['amount']
        elif 'amount' in args and type(args['amount']) == list:
            new = []
            for ia in args['amount']:
                if ia != '' and ia.lower() != 'outputAmount'.lower() and ia != 'all' and ia != 'half':
                    if ia.lower() not in message.lower():
                        continue
                    else:
                        try:
                            y = float(ia.replace('%','').replace('$','').replace(',',''))
                        except:
                            continue
                new.append(ia)
            if len(new) > 1:
                args['amount'] = new
            elif len(new) == 1:
                args['amount'] = new[0]
            else:
                del args['amount']
        if 'inputAmount' in args and 'inputToken' in args and type(args['inputAmount']) == list and type(args['inputToken']) == list:
            assert len(args['inputAmount']) == len(args['inputToken'])
            zipped = set(zip(args['inputAmount'], args['inputToken']))
            newa = []
            newt = []
            for z in zipped:
                newa.append(z[0])
                newt.append(z[1])
            assert len(newa) == len(newt)
            if len(newa) > 1:
                args['inputAmount'] = newa
                args['inputToken'] = newt
            elif len(newa) == 1:
                args['inputAmount'] = newa[0]
                args['inputToken'] = newt[0]
            else:
                raise Exception(f"{args['inputAmount']} {args['inputToken']}")
        if 'inputAmount' in args and 'inputToken' in args and type(args['inputAmount']) == list and type(args['inputToken']) == str:
            args['inputAmount'] = args['inputAmount'][0]
        if 'inputAmount' in args and 'inputToken' in args and type(args['inputAmount']) == list and type(args['inputToken']) == list:
            assert len(args['inputAmount']) == len(args['inputToken'])
            newt = []
            newa = []
            for t, a in zip(args['inputToken'], args['inputAmount']):
                if t == a and t == 'all':
                    continue
                else:
                    newt.append(t)
                    newa.append(a)
            assert len(newa) == len(newt)
            if len(newa) > 1:
                args['inputAmount'] = newa
                args['inputToken'] = newt
            elif len(newa) == 1:
                args['inputAmount'] = newa[0]
                args['inputToken'] = newt[0]
            else:
                raise Exception(f"{args['inputAmount']} {args['inputToken']}")
        if 'amount' in args and 'token' in args and type(args['amount']) == list and type(args['token']) == list:
            assert len(args['amount']) == len(args['token'])
            zipped = set(zip(args['amount'], args['token']))
            newa = []
            newt = []
            for z in zipped:
                newa.append(z[0])
                newt.append(z[1])
            assert len(newa) == len(newt)
            if len(newa) > 1:
                args['amount'] = newa
                args['token'] = newt
            elif len(newa) == 1:
                args['amount'] = newa[0]
                args['token'] = newt[0]
            else:
                raise Exception(f"{args['amount']} {args['token']}")
        if 'amount' in args and 'token' in args and type(args['amount']) == list and type(args['token']) == str:
            args['amount'] = args['amount'][0]
        if 'amount' in args and 'token' in args and type(args['amount']) == list and type(args['token']) == list:
            assert len(args['amount']) == len(args['token'])
            newt = []
            newa = []
            for t, a in zip(args['token'], args['amount']):
                if t == a and t == 'all':
                    continue
                else:
                    newt.append(t)
                    newa.append(a)
            assert len(newa) == len(newt)
            if len(newa) > 1:
                args['amount'] = newa
                args['token'] = newt
            elif len(newa) == 1:
                args['amount'] = newa[0]
                args['token'] = newt[0]
            else:
                raise Exception(f"{args['amount']} {args['token']}")
        if (name == 'swap' or name == 'bridge') and 'protocolName' in args and len(signs) > 0 and 'protocolName' in signs[-1]['args'] and signs[-1]['name'] not in ['swap', 'bridge'] and args['protocolName'] == signs[-1]['args']['protocolName']:
            del args['protocolName']
        if 'poolName' in args and 'protocolName' not in args:
            del args['poolName']
        if 'token' in args and type(args['token']) == str and args['token'] in chains:
            if name != 'bridge':
                if 'chainName' not in args:
                    args['chainName'] = args['token']
                del args['token']
            if name == 'bridge':
                if 'sourceChainName' not in args:
                    args['sourceChainName'] = args['token']
                del args['token']
        if 'inputToken' in args and type(args['inputToken']) == str and args['inputToken'] in chains:
            if name != 'bridge':
                if 'chainName' not in args:
                    args['chainName'] = args['inputToken']
                del args['inputToken']
            if name == 'bridge':
                if 'sourceChainName' not in args:
                    args['sourceChainName'] = args['inputToken']
                del args['inputToken']
        if name in ['long', 'short'] and 'inputToken' in args and ('outputToken' not in args or args['outputToken'] == ''):
            args['outputToken'] = args['inputToken']
        if 'token' in args and type(args['token']) == str and 'poolName' in args and args['poolName'] + ' lp' == args['token']:
            del args['poolName']
        if ("amount_units" in args and ("token" not in args or ("token" in args and args["token"] == "")) and args["amount_units"].lower() in tokens):
            args["token"] = args["amount_units"]
            del args["amount_units"]
        if 'comparator' in args:
            unders = ['below', 'is under', 'lt', 'lte', 'less than', 'less_than', 'lessThan', 'lessthan', '<', 'goes below', 'sub', 'is at or below']
            overs = ['above', 'is over', 'gt', 'gte', 'greater than', 'greater_than', 'greaterThan', 'greaterthan', '>', 'goes above', 'is at or above']
            equals = ['equals', 'hit', 'hits', '=', 'is at', 'reaches']
            notequals = ['notequals']
            if args['comparator'] in unders:
                args['comparator'] = '<='
            if args['comparator'] in overs:
                args['comparator'] = '>='
            if args['comparator'] in equals or args['comparator'] == 'is' or args['comparator'] == 'at':
                args['comparator'] = '=='
            if args['comparator'] in notequals:
                args['comparator'] = '!='
            if args['comparator'] in ['dips', 'decrease', 'decreases', 'decrease by', 'decreases by']:
                args['comparator'] = '<='
                if 'value' in args and args['value'][0] != '-':
                    args['value'] = '-' + args['value']
        if 'value' in args:
            if 'price' in args['value']:
                args['value'] = args['value'].replace(" price ", "")
                args['value'] = args['value'].replace("price ", "")
                args['value'] = args['value'].replace("_price", "")
                args['value'] = args['value'].replace("price_of_", "")
                args['value'] = args['value'].replace("price", "")
                args['type'] = 'price'
            elif 'market cap' in args['value'] or 'market_cap' in args['value']:
                args['value'] = args['value'].replace(" market cap", "")
                args['value'] = args['value'].replace("_market cap", "")
                args['value'] = args['value'].replace("_market_cap", "")
                args['value'] = args['value'].replace("market_cap_of_", "")
                args['value'] = args['value'].replace("market cap", "")
                args['value'] = args['value'].replace("market_cap", "")
                args['type'] = 'market cap'
            split = [xx for xx in args['value'].split(' ') if xx != '']
            if len(split) == 2:
                args['value'] = split[0]
                args['value_token'] = split[1]
                if args['value_token'] == 'dollars' or args['value_token'] == 'usd':
                    del args['value_token']
        if 'slippage' in args and (name != 'swap' or 'slippage' not in message.lower()):
            del args['slippage']
        if 'subject' in args:
            settype = False
            if 'type' not in args:
                settype = True
            if 'price' in args['subject']:
                args['subject'] = args['subject'].replace(" price", "")
                args['subject'] = args['subject'].replace("_market_price", "")
                args['subject'] = args['subject'].replace("_price", "")
                args['subject'] = args['subject'].replace("price_of_", "")
                args['subject'] = args['subject'].replace("price", "")
                if args['subject'] == '':
                    args['subject'] = 'outputToken'
                if settype:
                    args['type'] = 'price'
            elif 'balance' in args['subject']:
                args['subject'] = args['subject'].replace(" balance", "")
                args['subject'] = args['subject'].replace("_balance", "")
                args['subject'] = args['subject'].replace("balance_of_", "")
                args['subject'] = args['subject'].replace("balance", "")
                if args['subject'] == '':
                    args['subject'] = 'outputToken'
                if settype:
                    args['type'] = 'balance'
            elif 'market cap' in args['subject'] or 'market_cap' in args['subject']:
                args['subject'] = args['subject'].replace(" market cap", "")
                args['subject'] = args['subject'].replace("_market cap", "")
                args['subject'] = args['subject'].replace("_market_cap", "")
                args['subject'] = args['subject'].replace("market_cap_of_", "")
                args['subject'] = args['subject'].replace("market cap", "")
                args['subject'] = args['subject'].replace("market_cap", "")
                if args['subject'] == '':
                    args['subject'] = 'outputToken'
                if settype:
                    args['type'] = 'market cap'
            elif 'gas' in args['subject']:
                if settype:
                    args['type'] = 'gas'
            elif 'apy' in args['subject']:
                if settype:
                    args['type'] = 'yield'
            elif 'ltv' in args['subject']:
                if settype:
                    args['type'] = 'ltv'
            else:
                if settype:
                    args['type'] = 'price'
            if 'rewards' in args['subject'] and args['type'] == 'price':
                if settype:
                    args['type'] = 'balance'
            if args['subject'] == '':
                args['subject'] = 'outputToken'
            # args['subject'] = re.sub(rxtk + '_' + rxtk2, r'\g<tkn>/\g<tkn2>', args['subject'])
        if 'start_time' in args and 'end_time' in args and args['end_time'] == args['start_time']:
            del args['end_time']
        if 'start_time' in args and 'end_time' in args and args['end_time'] == 'tomorrow':
            if args['start_time'] != 'now':
                args['start_time'] = args['start_time']+' tomorrow'
            del args['end_time']
        if 'recurrence' in args and args['recurrence'] == None:
            del args['recurrence']
        addcond = {}
        if name != 'condition' and ('subject' in args or 'comparator' in args or 'value' in args or 'value_token' in args or 'type' in args):
            if 'subject' in args and args['subject'] == 'time':
                addcond = {"name": "time", "args": {}}
                if "operator" in args:
                    addcond["args"]["operator"] = args["operator"]
                if "value" in args:
                    addcond["args"]["start_time"] = args["value"]
                if "recurrence" in args:
                    addcond["args"]["recurrence"] = args["recurrence"]
                if "end_time" in args:
                    addcond["args"]["end_time"] = args["end_time"]
            else:
                addcond = {"name": "condition", "args": {}}
                if 'operator' in args:
                    addcond["args"]["operator"] = args["operator"]
                if "subject" in args:
                    addcond["args"]["subject"] = args["subject"]
                if "comparator" in args:
                    addcond["args"]["comparator"]: args["comparator"]
                if "value" in args:
                    addcond["args"]["value"]: args["value"]
                if "value_token" in args:
                    addcond['args']['value_token'] = args['value_token']
                if "type" in args:
                    addcond['args']['type'] = args['type']
            if 'operator' in args:
                del args['operator']
            if 'subject' in args:
                del args['subject']
            if 'comparator' in args:
                del args['comparator']
            if 'value' in args:
                del args['value']
            if 'value_token' in args:
                del args['value_token']
            if 'start_time' in args:
                del args['start_time']
            if 'recurrence' in args:
                del args['recurrence']
            if 'end_time' in args:
                del args['end_time']
            if 'type' in args:
                del args['type']
            if 'value_token' in addcond and (addcond['value_token'] == 'dollars' or addcond['value_token'] == 'usd'):
                del addcond['args']['value_token']
        addtime = {}
        if name != 'time' and ('start_time' in args or 'recurrence' in args or 'end_time' in args):
            addtime = {"name": "time", "args": {}}
            if 'operator' in args:
                addtime["args"]["operator"] = args["operator"]
                del args['operator']
            if 'start_time' in args:
                addtime["args"]["start_time"] = args["start_time"]
                del args['start_time']
            if 'end_time' in args:
                if ('start_time' in args and args['start_time'] != args['end_time']) or 'start_time' not in args:
                    addtime["args"]["end_time"] = args["end_time"]
                del args['end_time']
            if 'recurrence' in args:
                if args['recurrence'] != None:
                    addtime["args"]["recurrence"] = args["recurrence"]
                del args['recurrence']
        x = {"name": name, "args": args}
        if x in signs:
            continue
        signs.append(x)
        if addcond != {} and list(addcond["args"].keys()) != ["operator"]:
            signs.append(addcond)
        if addtime != {} and addtime != {"name": "time", "args": {"start_time": "now"}}:
            signs.append(addtime)
    if len(signs) > 0:
        processed = []
        for s in range(len(signs)-1):
            cur = signs[s]
            nex = signs[s+1]
            # print(cur, nex)
            if (cur['name'] == 'swap' or cur['name'] == 'bridge') and 'protocolName' in cur['args'] and 'protocolName' in nex['args'] and nex['name'] not in ['swap', 'bridge'] and cur['args']['protocolName'] == nex['args']['protocolName']:
                if not (s > 0 and signs[s-1]['name'] in ['swap', 'bridge'] and 'protocolName' in signs[s-1]['args'] and signs[s-1]['args']['protocolName'] == cur['args']['protocolName']):
                    del cur['args']['protocolName']
            if 'poolName' in cur['args'] and 'protocolName' not in cur['args']:
                del cur['args']['poolName']
            prevnames = []
            for ss in range(s):
                prevnames.append(signs[ss]['name'])
            prevnames = set(prevnames)
            if (prevnames == {"condition", "time"} or prevnames == {"time", "condition"} or prevnames == {"time"} or prevnames == {"condition"} or s == 0) and cur['name'] not in ['swap', 'bridge', 'condition', 'time', 'transfer'] and ('protocolName' not in cur['args'] or cur['args']['protocolName'] == ''):
                for ss in range(s, len(signs)-1):
                    if 'protocolName' in signs[ss]['args']:
                        cur['args']['protocolName'] = signs[ss]['args']['protocolName']
            processed.append(cur)
        processed.append(signs[-1])
        assert len(signs) == len(processed), (signs, processed)
        signs = processed
        processed = []
        first = -1
        for f in range(len(signs)):
            if signs[f]['name'] not in ['condition', 'time']:
                first = f
                break
            processed.append(signs[f])
        if first != -1:
            x = signs[first]
            if x['name'] not in ['swap', 'bridge', 'transfer', 'condition', 'time'] and ('protocolName' not in x['args'] or x['args']['protocolName'] == ''):
                x['args']['protocolName'] = 'all'
            processed.append(x)
            for s in range(first+1, len(signs)):
                cur = signs[s]
                if cur['name'] not in ['swap', 'bridge', 'transfer', 'condition', 'time'] and ('protocolName' not in cur['args'] or cur['args']['protocolName'] == ''):
                    for ss in range(s-1, -1, -1):
                        if signs[ss]['name'] not in ['swap', 'bridge', 'transfer', 'condition', 'time'] and 'protocolName' in signs[ss]['args']:
                            cur['args']['protocolName'] = signs[ss]['args']['protocolName']
                            break
                processed.append(cur)
        assert len(signs) == len(processed), (signs, processed)
        signs = processed
        processed = []
        for ix, s in enumerate(signs):
            if ix == 0 and 'operator' in s['args'] and len(signs) > 1 and signs[ix+1]['name'] not in ['condition', 'time']:
                del s['args']['operator']
            if ix == len(signs)-1 and 'operator' in s['args'] and len(signs) > 1 and signs[ix-1]['name'] not in ['condition', 'time']:
                del s['args']['operator']
            if ix != 0 and ix != len(signs)-1 and 'operator' in s['args'] and len(signs) > 2 and signs[ix+1]['name'] not in ['condition', 'time'] and signs[ix-1]['name'] not in ['condition', 'time']:
                del s['args']['operator']
            processed.append(s)
        assert len(signs) == len(processed), (signs, processed)
        signs = processed
        processed = []
        for ix in range(len(signs)):
            s = signs[ix]
            if 'inputAmount' in s['args'] and 'outputToken' in s['args']:
                target = s['args']['outputToken']
                msglist = message.split(' ')
                idcs = [i-1 for i, x in enumerate(msglist) if x == target]
                before = None
                for idc in idcs:
                    if msglist[idc].isdigit():
                        before = msglist[idc]
                        break
                if before:
                    try:
                        y = float(before.replace('%','').replace('$','').replace(',',''))
                        if y == float(s['args']['inputAmount']):
                            s['args']['outputAmount'] = before
                            del s['args']['inputAmount']
                    except:
                        pass
            processed.append(s)
        assert len(signs) == len(processed), (signs, processed)
        signs = processed
        processed = []
        for ix in range(len(signs)):
            s = signs[ix]
            if ix > 0 and 'subject' in s['args'] and s['args']['subject'].lower() in ['outputtoken', 'outputamount']:
                if "side" in signs[ix-1]['args']:
                    if signs[ix-1]['args']['side'] == 'sell':
                        s['args']['subject'] = signs[ix-1]['args']['inputToken']
                    if signs[ix-1]['args']['side'] == 'buy':
                        s['args']['subject'] = signs[ix-1]['args']['outputToken']
                else:
                    if 'inputAmount' in signs[ix-1]['args'] and 'inputToken' in signs[ix-1]['args']:
                        if signs[ix-1]['args']['inputToken'] in ['usdc', 'usdt', 'dai'] and s['args']['type'] == 'price' and (float(s['args']['value']) < 0.89 or float(s['args']['value']) > 1.11) and 'outputToken' in signs[ix-1]['args']:
                            s['args']['subject'] = signs[ix-1]['args']['outputToken']
                        else:
                            s['args']['subject'] = signs[ix-1]['args']['inputToken']
                    elif 'outputAmount' in signs[ix-1]['args'] and 'outputToken' in signs[ix-1]['args']:
                        s['args']['subject'] = signs[ix-1]['args']['outputToken']
            if len(signs) > 1 and ix == 0 and 'subject' in s['args'] and s['args']['subject'].lower() in ['outputtoken', 'outputamount']:
                if "side" in signs[ix+1]['args']:
                    if signs[ix-1]['args']['side'] == 'sell':
                        s['args']['subject'] = signs[ix+1]['args']['inputToken']
                    if signs[ix-1]['args']['side'] == 'buy':
                        s['args']['subject'] = signs[ix+1]['args']['outputToken']
                else:
                    if 'inputAmount' in signs[ix+1]['args'] and 'inputToken' in signs[ix+1]['args']:
                        if signs[ix+1]['args']['inputToken'] in ['usdc', 'usdt', 'dai'] and s['args']['type'] == 'price' and (float(s['args']['value']) < 0.89 or float(s['args']['value']) > 1.11) and 'outputToken' in signs[ix+1]['args']:
                            s['args']['subject'] = signs[ix+1]['args']['outputToken']
                        else:
                            s['args']['subject'] = signs[ix+1]['args']['inputToken']
                    elif 'outputAmount' in signs[ix+1]['args'] and 'outputToken' in signs[ix+1]['args']:
                        s['args']['subject'] = signs[ix+1]['args']['outputToken']
            for iy in range(ix+1, len(signs)):
                if 'operator' in signs[iy]['args']:
                    if 'subject' in signs[iy]['args'] and signs[iy]['args']['subject'].lower() in ['outputtoken', 'outputamount']:
                        ss = signs[iy]
                        ss['args']['subject'] = s['args']['subject']
                else:
                    break
            processed.append(s)
        assert len(signs) == len(processed), (signs, processed)
        signs = processed
        processed = []
        for ix, s in enumerate(signs):
            if "side" in s['args']:
                del s['args']['side']
            processed.append(s)
        assert len(signs) == len(processed), (signs, processed)
        signs = processed
        processed = []
        count = 0
        for ix, s in enumerate(signs):
            if ix != 0 and "protocolName" not in signs[ix-1]["args"]:
                if s["name"] == "transfer" and "recipient" in s['args']:
                    x = s['args']['recipient'].replace("_protocol", "").replace("protocol", "").replace("_address", "").replace("address", "")
                    if x in protocols:
                        processed[-1]['args']['protocolName'] = x
                        count += 1
                        continue
            processed.append(s)
        assert len(signs) == len(processed) + count, (signs, processed)
        signs = processed
        processed = [signs[0]]
        count = 0
        for ix in range(1, len(signs)):
            if signs[ix]['name'] == signs[ix-1]['name']:
                if 'amount' in signs[ix]['args'] and 'token' in signs[ix]['args'] and 'amount' in signs[ix-1]['args'] and 'token' in signs[ix-1]['args']:
                    if signs[ix]['args']['amount'] == signs[ix-1]['args']['amount'] and signs[ix]['args']['amount'] == 'all':
                        if signs[ix]['args']['token'] == signs[ix-1]['args']['token']:
                            count += 1
                            processed.pop()
            processed.append(signs[ix])
        assert len(signs) == len(processed) + count, (signs, processed)
        signs = processed
        processed = []
        for ix in range(len(signs)-1):
            if 'destinationChainName' in signs[ix+1]['args'] and 'chainName' in signs[ix]['args']:
                if signs[ix]['args']['chainName'] == signs[ix+1]['args']['destinationChainName']:
                    del signs[ix]['args']['chainName']
            processed.append(signs[ix])
        processed.append(signs[-1])
        assert len(signs) == len(processed), (signs, processed)
        signs = processed
        processed = []
        for ix in range(len(signs)):
            if ix <= len(signs) // 2 and 'recurrence' in signs[ix]['args']:
                if 'tomorrow' in message[:len(message)//2] and 'start_time' in signs[ix]['args'] and 'tomorrow' not in signs[ix]['args']['start_time']:
                    if signs[ix]['args']['start_time'] != 'now':
                        signs[ix]['args']['start_time'] = signs[ix]['args']['start_time'] + ' tomorrow'
                    del signs[ix]['args']['recurrence']
                if 'today' in message[:len(message)//2] and 'start_time' in signs[ix]['args'] and 'today' not in signs[ix]['args']['start_time']:
                    signs[ix]['args']['start_time'] = signs[ix]['args']['start_time'] + ' today'
                    del signs[ix]['args']['recurrence']
            if ix >= len(signs) // 2 and 'recurrence' in signs[ix]['args']:
                if 'tomorrow' in message[len(message)//2:] and 'start_time' in signs[ix]['args'] and 'tomorrow' not in signs[ix]['args']['start_time']:
                    if signs[ix]['args']['start_time'] != 'now':
                        signs[ix]['args']['start_time'] = signs[ix]['args']['start_time'] + ' tomorrow'
                    del signs[ix]['args']['recurrence']
                if 'today' in message[len(message)//2:] and 'start_time' in signs[ix]['args'] and 'today' not in signs[ix]['args']['start_time']:
                    signs[ix]['args']['start_time'] = signs[ix]['args']['start_time'] + ' today'
                    del signs[ix]['args']['recurrence']
            processed.append(signs[ix])
        assert len(signs) == len(processed), (signs, processed)
        signs = processed
    for s in signs:
        print(s['name'], s['args'])
    print('')
    # track_response(message_id, signs, test=True)
    if save > 0:
        num = save
        with open(f'test/res{num}.json', 'r') as f:
            sofar = json.load(f)
        lk = {x.lower(): x for x in list(sofar.keys())}
        if om in lk:
            sofar[lk[om]] = signs
        else:
            sofar[om] = signs
        with open(f'test/res{num}.json', 'w') as f:
            json.dump(sofar, f)
    return results, response


# examples that our model should achieve 100% accuracy on
def suite(step=0, prompt=1, save=0):
    if step == -3:
        perform_message("When $TOK is at 12k market cap, buy $20 worth of $TOK", prompt, save)
        perform_message("Trade 10000 USDC for $RLB when price reaches 10 cents", prompt, save)
        perform_message("buy xyz if its market cap falls below 50mn over the next 7 days", prompt, save)
        perform_message("dump all positions to usdc", prompt, save)
        perform_message("repay my borrow position on dolomite when borrow apy rises above 69%", prompt, save) # "Unwind my borrow position on Dolomite when borrow APY rises above 69%", prompt, save)
        perform_message("Long btc with 3x leverage at 6pm GMT today", prompt, save)
        perform_message("once eth balance hits X, buy Y of $token when gas is below z", prompt, save)
        perform_message("swap 0.25eth for *contract-adress using 5% slippage", prompt, save)
        perform_message("each monday, claim my vested tokens, and stake them in Jones jUSDC pool", prompt, save) # "Each monday, claim my vested tokens, and stake them in xyz pool", prompt, save)
        perform_message("do a swap 10 usdt to zkusd on syncswap on all my accounts, at randomized time(random range is 8 hours)", prompt, save)
        perform_message("take my 1000 usdc, convert it into eth, deposit it into my prologue nft on spice, then borrow 60%", prompt, save)
        perform_message("Using 2 ETH buy USDC, USDT, and DAI, then deposit into Curve tricrypto pool", prompt, save)
        perform_message("Deposit 100 ARB into Plutus, stake LP for PLS, then lock PLS", prompt, save)
        perform_message("get out of my OHM and into stETH, bridge everything to arbitrum.", prompt, save)
        perform_message("Claim and compound rewards from GND protocol", prompt, save)
        perform_message("sell everything from multiple chains into jusdc", prompt, save)
        perform_message("buy $eth with 7500 $usdc when $eth is $1400 and sell it all at $1600", prompt, save)
        perform_message("Once my Plutus rewards hit 2 ETH, claim rewards and transfer to person.eth", prompt, save)
        perform_message("on pancakeswap and zksync era network, swap all of my USDC for WETH, and reverse this process", prompt, save)
        perform_message("swap 2.5 ETH for SIS and and 2.5 ETH for USDC (on symbiosis finance & the zkSync network) and then swap both back to ETH", prompt, save)
        perform_message("on mav protocol, swap 1 ETH for LUSD, 1 ETH for USDC, and 1 ETH for MAV, LP into USDC-LUSD pair", prompt, save)
        perform_message("claim all rewards, swap everything into ETH, bridge to mainnet", prompt, save)
        perform_message("send 0.05 eth to 0x6955e7216e8d9d2ab2ca5ca5e31ccf7307e9d59f when gas is < 10", prompt, save)
        perform_message("sell all my $TEMPLE for USDC when TEMPLE/USDC > 1.03. afterwards swap the USDC to ETH when gas is <6", prompt, save)
        perform_message("swap all my merit circle, dai, fxs, for ETH. then brdige all the ETH as well as my 12,227 USDC position over to arbitrum when gas is <10", prompt, save)
        perform_message("Claim liveTHE rewards once their balance is greater than 10$ and deposit them back into the thena liveTHE pool", prompt, save)
        perform_message("When camelot rewards balance hits $100. Claim and convert all rewards to weth. Then deposit in spice vault.", prompt, save)
        perform_message("at 10am, swap 100 usdc to eth, if gas below 50", prompt, save)
        perform_message("LP 2 ETH into balancer and compound the rewards every 3 days", prompt, save)
        perform_message("Borrow 1000 usdc.e when usdc.e is above $1.005 and swap to usdt", prompt, save)
        perform_message("trade 1 baby bear for eth then swap that eth for usdc and send that usdc over to the arbitrum network with hop exchange", prompt, save)
    if step == -2:
        perform_message("Buy JONES with half my ETH, deposit into the ETH-JONES pool on Sushi, then trade LP for plsJones", prompt, save)
        perform_message("Swap half of all my tokens to ETH and transfer to niyant.eth on mainnet", prompt, save)
        perform_message("Bridge 1 ETH to Base, swap half to USDC, deposit into Kyber eth-usdc pool", prompt, save)
        perform_message("Withdraw half the liquidity from my Dolomite USDC-USDT position", prompt, save)
    if step == -1:
        perform_message("Bridge 1 WETH from Base to Ethereum and deposit in Aave", prompt, save)
        perform_message("Swap 10 ETH for USDC when ETH is below 1600", prompt, save)
        perform_message("Bridge all my tokens on Canto to Ethereum", prompt, save)
        perform_message("At 10am tomorrow, transfer 200 USDC to 0x2B605C2a76EE3F08a48b4b4a9d7D4dAD3Ed46bf3", prompt, save)
        perform_message("Harvest my MMF yield farms and automatically stake MMF every day at 8am", prompt, save)
        perform_message("Vote on Solidly every Wednesday and claim Solidly rewards every Thursday", prompt, save)
        perform_message("Vote, harvest, and restake all my positions every day", prompt, save)
        perform_message("Withdraw 100 USDC from JonesDAO, bridge to Ethereum, and deposit it into Yearn", prompt, save)
        perform_message("Claim and restake my Chronos position every week on Monday", prompt, save)
        perform_message("Claim STG from my Stargate positions, swap to WETH, and deposit back into Stargate", prompt, save)
        perform_message("Buy JONES with half my ETH, deposit into the ETH-JONES pool on Sushi, then trade LP for plsJones", prompt, save)
        perform_message("Swap 5000 USDC for ETH on Sushiswap on Ethereum, bridge to Base, sell ETH for USDC on KyberSwap, bridge USDC back to mainnet", prompt, save)
        perform_message("Harvest all my rewards on Arbitrum and buy ETH", prompt, save)
        perform_message("Lend 5 ETH, borrow 100 PT, then deposit 100 PT and 100 GLP into the PT-GLP pool on Pendle", prompt, save)
        perform_message("Claim Redacted rewards and relock BTRFLY", prompt, save)
        perform_message("Withdraw all my USDC from Aave and deposit into Compound", prompt, save)
        perform_message("Unstake all my plsDPX and sell it for DPX if the price of plsDPX/DPX < 0.95", prompt, save)
        perform_message("Swap half of all my tokens to ETH and transfer to niyant.eth on mainnet", prompt, save)
        perform_message("Deposit 50 USDC and 50 USDT into DODO Finance USDC-USDT pool, then every Friday claim DODO and swap to usdt", prompt, save) # FAIL # "You LP USDC-USDT and earn DODO at 7-8% APY which you can dump for stables"
        perform_message("when my ETH balance hits 1, buy 0.5 ETH worth of SAINT once the price of SAINT/ETH is under 20 and gas under 15", prompt, save) # FAIL 
        perform_message("Stake STG on stargate, then every Friday claim and restake rewards", prompt, save) # perform_message("Stake STG on stargate, every Friday claim and restake rewards every week on Friday", prompt, save)
    if step == 0 or step == 1:
        perform_message("Swap 1 ETH for USDC", prompt, save)
        perform_message("Swap 1 ETH for USDC on Uniswap", prompt, save)
        perform_message("Bridge 1 USDT from Base to Arbitrum", prompt, save)
        perform_message("Bridge 1 USDT from Base to Arbitrum on Hop Protocol", prompt, save)
        perform_message("Transfer 10 DAI to niyant.eth", prompt, save)
        perform_message("Swap 1 ETH for USDC on Ethereum then bridge to Arbitrum", prompt, save)
        perform_message("Bridge 1 ETH from Ethereum to Optimism then buy USDC", prompt, save)
        perform_message("Bridge 1 WETH from Base to Ethereum and deposit in Aave", prompt, save)
    if step == 0 or step == 2:
        perform_message("Swap 10 ETH for USDC when gas is below 20", prompt, save)
        perform_message("Swap 10 ETH for USDC when ETH is below 1600", prompt, save)
        perform_message("Swap 10 ETH for USDC in twelve hours", prompt, save)
        perform_message("Swap 10 ETH for USDC at 5pm", prompt, save)
        perform_message("Swap 10 ETH for USDC in twelve hours, repeating every twelve hours", prompt, save)
        perform_message("Swap 10 ETH for USDC at 5pm, repeating every 1 hour", prompt, save)
    if step == 0 or step == 3:
        perform_message("Deposit all my WETH into Aave", prompt, save)
        perform_message("Swap all my WETH into USDC", prompt, save)
        perform_message("Buy USDT with all my WETH", prompt, save)
        perform_message("Bridge all my WETH to Base", prompt, save)
        perform_message("Withdraw 0.1 ETH from Compound and buy OP", prompt, save)
        perform_message("Bridge 3 ETH to Avalanche and buy OHM", prompt, save)
        perform_message("Use 3 ETH to buy OHM on Avalanche", prompt, save)
        perform_message("Buy GRAIL with 4 WETH", prompt, save)
        perform_message("Bridge all my tokens on Canto to Ethereum", prompt, save)
    if step == 0 or step == 4:
        perform_message("Open a short trade on Kwenta on BTC with 3 ETH with 3x leverage", prompt, save) # "Hey can you get some eth, convert it, deposit in kwenta, open a short trade with this leverage"
        perform_message("Withdraw from all my positions, convert to WETH, and bridge to Arbitrum", prompt, save)
        perform_message("swap eth for usdt, swap usdc for usdt, bridge usdt to arbitrum", prompt, save)
        perform_message("When gas is below 10, deposit 100 USDC into Morpho", prompt, save)
        perform_message("At 10am tomorrow, transfer 200 USDC to 0x2B605C2a76EE3F08a48b4b4a9d7D4dAD3Ed46bf3", prompt, save)
        perform_message("Stake 10 ETH on Rocket Pool", prompt, save)
        perform_message("Harvest all my positions on Arbitrum", prompt, save)
        perform_message("Swap all my tokens on Optimism to WETH and bridge to Arbitrum", prompt, save)
    if step == 0 or step == 5:
        perform_message("Swap 1 ETH to USDC, bridge to Arbitrum, deposit into JonesDAO, then deposit LP into Rodeo", prompt, save)
        perform_message("Bridge 1 ETH to Base, swap half to USDC, deposit into Kyber eth-usdc pool", prompt, save)
        perform_message("Harvest my MMF yield farms and automatically stake MMF every day at 8am", prompt, save)
        perform_message("Harvest my positions every Wednesday", prompt, save) # FAIL
        perform_message("3x leverage long GLP with 1000 USDC on GMX and swap 1000 USDC into UNI", prompt, save) # "Ok i want to 3x leverage GLP and zap out of USDC into some other asset thats support on Dolomite"
        perform_message("Swap 500 DAI for WBTC every day for a month when gas is less than 30", prompt, save) # perform_message("carry out x no of swap on the dapp daily for 1 month when gas is less than 30", prompt, save)
        perform_message("Claim and restake rewards from all my positions every Monday", prompt, save) # perform_message("Claim LP staking rewards/airdrops to add back to the LP", prompt, save)
        perform_message("Bridge 200 USDT from Ethereum to Base and buy PEPE", prompt, save) # perform_message("Bridge from mainnet and long a coin in one sweep.", prompt, save)
    if step == 0 or step == 6:
        perform_message("harvesting on camelot", prompt, save)
        perform_message("Using 2 ETH buy USDC, USDT, and DAI, then deposit into Curve tricrypto pool", prompt, save) # perform_message("buy x amount usdc , x amount usdt, x amount DAI, and stake in trycrypto curve pool", prompt, save)
        perform_message("Vote on my THENA position every week on Wednesday", prompt, save)
        perform_message("Deposit 100 ARB into Plutus, stake LP for PLS, then lock PLS", prompt, save) # perform_message("Deposit 100 ARB into Plutus and stake LP for PLS, then lock PLS", prompt, save)
        perform_message("withdraw position from trader joe", prompt, save)
        perform_message("Vote on Solidly every Wednesday and claim Solidly rewards every Thursday", prompt, save) # perform_message("voting solidly forks every week and claiming rewards of the same next day", prompt, save)
    if step == 0 or step == 7:
        perform_message("Borrow USDC from Compound and deposit into Aave", prompt, save) # FAIL
        perform_message("Borrow 1000 USDC from Compound and deposit into Aave", prompt, save)
        perform_message("Withdraw from all my positions on Ethereum and convert everything to ETH", prompt, save)
        perform_message("Vote, harvest, and restake all my positions every day", prompt, save)
        perform_message("Vote on all my positions every Sunday", prompt, save) # perform_message("Vote on all my positions once a week", prompt, save)
        perform_message("vote on the most optimal pair on solidly every wednesday at this time", prompt, save)
        perform_message("Harvest and restake all my positions every week", prompt, save)
    if step == 0 or step == 8:
        perform_message("Process rewards on Redacted Cartel, swap to WETH, and deposit into Blur, biweekly", prompt, save) # "process RLBTRFLY rewards bi weekly....then take the weth i receive and deposit into blur vault."
        perform_message("grab weekly rewards from ve(3,3) DEXes and convert them to ETH", prompt, save) # FAIL
        perform_message("Grab rewards from Balancer and convert to ETH every week", prompt, save) # FAIL
        perform_message("Bridge 1000 USDC from Ethereum to zkSync and deposit into PancakeSwap", prompt, save) # "spot a farm on defi lama and realize you dont have funds on lets say zkSync"
        perform_message("Withdraw 100 USDC from JonesDAO, bridge to Ethereum, and deposit it into Yearn", prompt, save) # "rebalancing pools and farms on different chains"
        perform_message("Claim and redeposit rewards on all my protocols every week on Wednesday", prompt, save) # "There are many different markets to claim rewards and reinvest as well for LP positions"
    if step == 0 or step == 9:
        perform_message("Buy BTC with 1 ETH every week", prompt, save) # FAIL # "Want to DCA every week"
        perform_message("Buy BTC with 1 ETH when BTC is at or below $25000 and sell 0.2 BTC for ETH when BTC is at or above $30000, forever", prompt, save) # "straddling buy / sell at specific prices" # "on Bitcoin, Buy at x price, Sell at y, Rinse and repeat for 48 hours" perform_message("arbitrage bot: buy btc on x and sell on y until price equilizes", prompt, save)
        perform_message("Claim and restake my Chronos position every week on Monday", prompt, save)
        perform_message("Bridge 4 USDT to Base", prompt, save)
        perform_message("Swap 3 ETH to USDC and deposit into the ETH-USDC pool on Dolomite", prompt, save)
        perform_message("Open a 2x ETH long on GMX with 1000 USDC", prompt, save)
        perform_message("Vote on my Thena position every Wednesday", prompt, save)
    if step == 0 or step == 10:
        perform_message("Withdraw 2 ETH from my ETH-USDC pool position on Camelot", prompt, save)
        perform_message("Claim STG from my Stargate positions, swap to WETH, and deposit back into Stargate", prompt, save)  # FAIL # "Compound my Stargate position"
        perform_message("for my pendle token, if it reaches $1.50, sell it for usdc. if it reaches $1.20, buy back with usdc", prompt, save)  # FAIL # perform_message("for my position in pendle, if it reaches $1.50, sell it. Buy back at $1.20", prompt, save)
        perform_message("Stake my ARB on Arbitrum", prompt, save) # "I want to stake my arb, please give me 3 options"
        perform_message("Harvest my Balancer position and stake the rewards", prompt, save)
        perform_message("Withdraw half the liquidity from my Dolomite USDC-USDT position", prompt, save)
    if step == 0 or step == 11:
        perform_message("Claim wETH-GRAIL LP rewards from Camelot and sell for USDC", prompt, save)  # FAIL # "You get staking rewards as stables and wETH-GRAIL LP.; You gotta exit the LP then sell each individually for whatever you want"
        perform_message("sell all my usdc for eth if usdc goes below $.95", prompt, save) # "conditional triggers (e.g. depeg on a stable)"
        perform_message("Buy JONES with half my ETH, deposit into the ETH-JONES pool on Sushi, then trade LP for plsJones", prompt, save) # "take eth and buy Jones then pair into lp position on sushi then take the lp token and trade it for plsjones"
    if step == 0 or step == 12:
        perform_message("Buy ETH with 1000 USDC on Uniswap on Ethereum, bridge to Optimism, and sell for USDC on Velodrome", prompt, save) # perform_message("faster arbitrage across chains", prompt, save)
        perform_message("Swap 5000 USDC for ETH on Sushiswap on Ethereum, bridge to Base, sell ETH for USDC on KyberSwap, bridge USDC back to mainnet", prompt, save)  # FAIL # perform_message("arbitrage process and he had to bridge + swap + send it everywhere + go to velodrome", prompt, save)
        perform_message("buy wbtc with eth on uniswap and sell it for eth on sushiswap", prompt, save)
        perform_message("buy wbtc with 1 eth on uniswap", prompt, save)
        perform_message("swap 1 eth for usdt", prompt, save)
        perform_message("swap XYZ for ABC on pancakeswap in 35 minutes", prompt, save)
        perform_message("swap XYZ for ABC on pancakeswap at 11 PM UST", prompt, save) # FAIL
    if step == 0 or step == 13:
        perform_message("Claim my Camelot rewards, swap to USDC, and deposit back into Camelot", prompt, save) # perform_message("claiming rewards and compounding g into the pool", prompt, save)
        perform_message("Buy WBTC with 1 ETH every Sunday", prompt, save) # perform_message("setting up DCA buying based on time and buy/sells on price levels", prompt, save)
        perform_message("Withdraw from my Lodestar position", prompt, save) # perform_message("withdrawing from LPs/staking", prompt, save)
        perform_message("Harvest all my rewards on Arbitrum and buy ETH", prompt, save) # perform_message("harvesting rewards, but seeking them and twapping into new tokens I want to accumulate", prompt, save)
    if step == 0 or step == 14:
        perform_message("Lend 5 ETH, borrow 100 PT, then deposit 100 PT and 100 GLP into the PT-GLP pool on Pendle", prompt, save) # perform_message("PT-GLP and money markets for PT-GLP; Something were lacking is looping strategies on PT; Would love to set up a prompt and have users execute operation in one go", prompt, save)
        perform_message("Deposit ETH into Pendle when APY is 10%", prompt, save)  # FAIL # perform_message("Creating limit orders with pendle - by PT when yield is at a specific level", prompt, save)
        perform_message("Lend 250 SMP and borrow 125 LMP on Pendle", prompt, save) # perform_message("Will short the short maturity and long the long maturity; Hes arbing the yield between the two pools", prompt, save)
        # perform_message("Buy PT with USDC on Pendle, then loop PT on Dolomite 5 times", prompt, save) # perform_message("Wants to loop with pendle to earn yield; have to go pendle to buy PT, then go to a market to borrow PT, then go back to pendle to buy PT", prompt, save)
    if step == 0 or step == 15:
        perform_message("Claim rewards from Camelot, swap rewards and GRAIL into xGRAIL, then deposit xGRAIL into Camelot", prompt, save) # FAIL # perform_message("claim rewards from camelot, 3 transactions to claim, plus two additional transactions to convert dust grail into xgrail then allocate to dividend", prompt, save)
        perform_message("Claim Redacted rewards and relock BTRFLY", prompt, save) # perform_message("relocked RL BTRFLY, CLAIM rewards also", prompt, save)
    if step == 0 or step == 16:
        perform_message("Withdraw all my USDC from Aave and deposit into Compound", prompt, save) # perform_message("Withdraw usdc from aave if compound usdc interest rate > aave.", prompt, save)
        perform_message("If bitcoin goes below 15k, buy eth", prompt, save) # perform_message("If bitcoin hits 15k, buy eth", prompt, save) # perform_message("marketbuy eth if bitcoin touches 15k", prompt, save)
        perform_message("Claim Stargate rewards, swap to ETH, redeposit", prompt, save) # perform_message("autocompounding any position", prompt, save)
        perform_message("Buy ETH with 5000 USDC. Sell ETH for USDC if the price goes below 1000 or above 3000", prompt, save)  # FAIL # perform_message("Buy ETH with 5000 USDC. Sell if ETH hits 1000 or 3000", prompt, save) # perform_message("setting up take profit/stop loss or optimizing pools", prompt, save)
    if step == 0 or step == 17:
        perform_message("Buy DPX with RDPX if the price of DPX/RDPX <= 0.8", prompt, save)  # FAIL # perform_message("Swing trading non-pooled pairs based on their ratio (dpx/rdpx)", prompt, save)
        perform_message("Unstake all my plsDPX and sell it for DPX if the price of plsDPX/DPX < 0.95", prompt, save)  # FAIL # perform_message("Unstaking and selling when ratio between a liquid derivative and the native asset hits certain ratio, being able to reverse that operation (say plsdpx on Plutus)", prompt, save)
        perform_message("Bridge 4 ETH from Arbitrum to Base and buy COIN when gas is under 12", prompt, save) # perform_message("Bridge from Arbitrum to Base and buy COIN when gas is under 12", prompt, save)
    if step == 0 or step == 18:
        perform_message("Swap all my tokens to ETH and buy ARB when gas is below 10", prompt, save)  # FAIL # perform_message("consolidate entire portfolio into ETH and get it onto arb when gas is low", prompt, save)
        perform_message("Swap all my tokens to ETH and transfer to niyant.eth on mainnet", prompt, save) # perform_message("turn everything into eth and send to preset addy on main.", prompt, save)
        perform_message("Swap half of all my tokens to ETH and transfer to niyant.eth on mainnet", prompt, save)
        perform_message("can you use my DAI to purchase sWeed", prompt, save)
        perform_message("Use DAI to purchase sWeed", prompt, save) # FAIL
    if step == 0 or step == 19:
        perform_message("Deposit 50 USDC and 50 USDT into DODO Finance USDC-USDT pool, then every Friday claim DODO and swap to usdt", prompt, save) # FAIL # "You LP USDC-USDT and earn DODO at 7-8% APY which you can dump for stables"
        perform_message("when my ETH balance hits 1, buy 0.5 ETH worth of SAINT once the price of SAINT/ETH is under 20 and gas under 15", prompt, save) # FAIL 
        perform_message("Stake STG on stargate, then every Friday claim and restake rewards", prompt, save) # perform_message("Stake STG on stargate, every Friday claim and restake rewards every week on Friday", prompt, save)
    if step == 0 or step == 20:
        perform_message("Swap 10 ETH for USDC when the ETH market cap is below 20", prompt, save)
        perform_message("Swap 10 ETH for USDC when the market cap of ETH is below 1600", prompt, save)
        perform_message("Swap 10 ETH for USDC when my ETH balance is below 1600", prompt, save)
        perform_message("When my camelot rewards balance is greater than 10 eth, swap to usdc", prompt, save)
        perform_message("Deposit 10 ETH into the Yearn yETH pool when the APY is 15%", prompt, save)
        perform_message("Deposit 10 ETH into the yETH pool on Yearn when APY is 15%", prompt, save)
        perform_message("buy 1 eth", prompt, save)
        perform_message("swap all my wbtc for usdt at 12pm tomorrow or if usdt price goes below $0.9", prompt, save)
        perform_message("swap xyz for abc when gas is below 14 and abc market cap is below 100 eth", prompt, save)
        perform_message("swap all my eth for usdc when gas is less than 20 or eth/usdt goes above 2000", prompt, save)
        perform_message("swap all my wbtc for usdt when my eth balance is greater than 2 or eth/dai goes above 2000", prompt, save)
        perform_message("Swap 1 eth for usdc with 2% slippage", prompt, save)
        perform_message("Swap 1 eth for usdc with max 3% slippage", prompt, save)
        perform_message("Bridge 1 eth for usdc with 2% slippage", prompt, save)
        perform_message("Bridge 1 eth for usdc then swap to dai with max 2% slippage", prompt, save)
        perform_message("swap all my usdt for dai", prompt, save)
        perform_message("swap all my usdt and usdc for dai", prompt, save)
        perform_message("Withdraw all my USDC and USDT from Rodeo, convert to ETH, and bridge all of it to mainnet", prompt, save) # FAIL # "Withdraw from Rodeo, convert to ETH, and bridge all of it to mainnet"
        perform_message("deposit 10 usdc and usdt into the uniswap usdc-usdt pool", prompt, save)
        perform_message("swap all my dai and half my usdt for usdc on curve", prompt, save)
        perform_message("Swap 10 ETH for USDC when ETH is below 1600", prompt, save)
        perform_message("Buy ETH with 1000 USDC when ETH/USDC is less than 2000", prompt, save)
        perform_message("Buy ETH with 1000 USDC when ETH/USDC price is less than 2000", prompt, save)
        perform_message("Buy ETH with 1000 USDC when the price of ETH/USDC is less than 2000", prompt, save)
        perform_message("when my USDC balance hits 3000, send it to arkham.eth", prompt, save)
        perform_message("bridge 1 eth from etheruem to arbitum", prompt, save)
        perform_message("deposit 1 gril into camlot", prompt, save)
        perform_message("buy ustd with 2 ETH", prompt, save)
        perform_message("when my dolomite rewards hit $2000, swap them for usdc", prompt, save)
        perform_message("when my dolomite rewards balance hits $2000, swap them for usdc", prompt, save)
        perform_message("Once my Plutus rewards hit 2 ETH, claim rewards and transfer to person.eth", prompt, save) # FAIL 
        perform_message("sell 2 eth", prompt, save)
        perform_message("buy 0.1 eth", prompt, save)
        perform_message("buy 10 usdc", prompt, save)
        perform_message("Buy btc with eth when it is at 20000")
        perform_message("when its at 20000, buy btc")
        perform_message("once btc hits 20000, sell all my btc at midnight", prompt, save)
        perform_message("at 2am, swap all my wbtc for eth if gas is less than 15", prompt, save)
        perform_message("swap 1 eth to usdc with 1.5% slippage when gas is less than 10", prompt, save)
    if step == 0 or step == 21:
        usermsgs = [
            "take my 1000 usdc, convert it into eth, deposit it into my prologue nft on spice, then borrow 60%",
            "bridge 5 eth from arbitrum to ethereum mainnet and long $pepe",
            "swap everything i own on eth mainnet to $eth and bridge it all to arbitrum",
            "withdraw 1 eth from my jonesdao position and ape $jesus",
            "swap by 2 $eth for $geth, convert to 1 $reth and 1 $steth, stake both on rocketpool",
            "deposit 0.33 ETH and 500 USDT in the ETH/USDT LP on Uniswap",
            "short $uni on october 16 at 12pm est",
            "buy $eth with 7500 $usdc when $eth is $1400 and sell it all at $1600",
            "Lend WETH as collateral on polylend, borrow WETH on polylend", # "Lend WETH as collateral on polylend, borrow WETH my WETH collateral with a max LTV of 80% and a borrow APY of -12.24%, farm with WETH on polylend",
            "Lend ETH as collateral on wing-finance which earns a Supply APY of 90.79%. Borrow USDT against your ETH collateral with a max LTV of 85% and a borrow APY of -1.55% (The interest you need to pay). Farm with USDT on paraspace-lending-v1 which earns 26.65%.",
            "swap all of my usdc to eth and eth to usdc on woofi on the zksync network", # "swap all of my usdc to eth and eth to usdc 20 times over on woofi on the zksync network",
            "on pancakeswap and zksync era network, swap all of my USDC for WETH, and reverse this process",
            "swap 2.5 ETH for SIS and and 2.5 ETH for USDC (on symbiosis finance & the zkSync network) and then swap both back to ETH",
            "on mav protocol, swap 1 ETH for LUSD, 1 ETH for USDC, and 1 ETH for MAV, LP into USDC-LUSD pair",
            "vote on the thena BNB/THE pool every wednesday at 8pm est",
            "deposit all of my wstETH in the Kyber axl-wstETH-wstETH pool on pendle",
            "claim all rewards, swap everything into ETH, bridge to mainnet",
            "claim RLBTRFLY rewards whenever available, convert to ETH and deposit into the SPICE blur vault",
            "swap 1 $eth to $usdc and then bridge it to Arbitrum",
            "bridge 2 $eth to arbitrum, swap $eth to $gmx, open an 0.5 $eth short position with 10x leverage with $eth market price, set stop loss at $1500 per $eth",
            "long $eth on arbitrum with 1 $eth",
            "send 0.5 $eth to bicep.eth",
            "withdraw all funds from the spice finance prologue vault, swap to eth, bridge all funds to arbitrum when gas is <10 swap everything except gas into steth", # "withdraw all funds from the Spice Finance Prologue Vault into ETH, bridge all funds to Arbitrum when gas is <10 swap everything except gas into stETH",
            "send 0.05 eth to 0x6955e7216e8d9d2ab2ca5ca5e31ccf7307e9d59f when gas is < 10",
            "send all my funds to 0x6955e7216e8d9d2ab2ca5ca5e31ccf7307e9d59f",
            "swap my ohm to steth, bridge everything to arbitrum", # "get out of my OHM and into stETH, bridge everything to arbitrum.",
            "bridge all funds from Canto to Ethereum, swap everything to ETH",
            "swap 0.22 ETH into stETH when gas is sub 6",
            "claim BLUR points whenever they release, swap them to ETH if gas is <40",
            "sell all my $TEMPLE for USDC when TEMPLE/USDC > 1.03. afterwards swap the USDC to ETH when gas is <6",
            "withdraw all my ETH from Yearn when gas is less than <6",
            "sell all my $DPI tokens for ETH when gas is less than 6",
            "swap all my merit circle, dai, fxs, for ETH. then brdige all the ETH as well as my 12,227 USDC position over to arbitrum when gas is <10",
            "Unvest 300000 Trove from Nitro Cartel and swap to USDC",
            "Revest remaining Trove at 16:00 UTC",
            "perform $XXX swap to $YYY using ___ (specific DEX) when gas is below __(given gwei)",
            "when gas is below x use defillama to swap ETH or USDC to x coin",
            "stake ETH on lido",
            "Bridge through stargate 100 usdc with cheapest fee's on any ETH L2",
            "claim my presale on this contract and do it weekly",
            "Trade 10000 USDC for $RLB when price reaches 10 cents",
            "when gas is under 20, bridge 1 eth to base and swap to $wig",
            "Claim liveTHE rewards once their balance is greater than 10$ and deposit them back into the thena liveTHE pool",
            "Claims LiveTHE rewards once the balance is greater than 10$ and swap to USDT",
            "Vote on thena on Wednesday at 2355 UTC",
            "When camelot rewards balance hits $100. Claim and convert all rewards to weth. Then deposit in spice vault.",
            "short Eth with usdc if it goes below or touches x price",
            "repay my borrow position on dolomite when borrow apy rises above 69%", # "Unwind my borrow position on Dolomite when borrow APY rises above 69%",
            "Pull my liquidity on Uniswap if price falls below X",
            "Sell half of asset X as soon as price hits $2",
            "When gas is below 8 bridge 0.1 eth to Zksync",
            "At exactly 10pm tomorrow buy \"Random CA\" with 40 gwei",
            "Swap 0.05 eth to usdt and send the swapped usdt to \"wallet address\"",
            "If USDR goes below 0.98, swap my USDR to DAI",
            "swap 0.25eth for *contract-adress using 5% slippage",
            "Long btc with 3x leverage at 6pm GMT today",
            "Buy 0.35 eth worth of $MOG and sell when mog/eth hits 0.7",
            "Stake 3 eth at lido for 2 weeks",
            "deposit 0.35 eth into aave, borrow $400 usdc and swap to $BITCOIN",
            "lock steth for 2 months",
            "Move all assets in active wallet to *x-adress",
            "when eth is below $1600, buy $500 usd worth each week",
            "vote for <pool> on <protocol> every Wednesday at 2355H (UTC)",
            "at 10am, swap 100 usdc to eth, if gas below 50",
            "bridge 2000 usdc to arbitrum, when gas <30",
            "whenever the eth price reaches $1,500, buy eth", # "Help me purchase ETH whenever the price reaches $1,500, use defillama's meta-dex aggregator and give me the best rate",
            "vote for the most profitable strategy on any >10m mcap -pool in the thena ve(3,3) voting pools. Do this at 11:55. At 12:05, collect the rewards of the previous voting epoch and exchange them for DOGE at market prices, using swap aggregator swap.defillama.com with 0.5% slippage.",
            "Buy 1 ETH when ETH price is $1550",
            "LP 2 ETH into balancer and compound the rewards every 3 days",
            "At exactly 19:05 UTC, bridge $50 ETH to Starknet ETH",
            "if ETH is over 1800, sell for USDC",
            "bridge [amount] ETH from Ethereum to Arbitrum using the most cost-effective method. then, convert it to WETH.",
            "Bridge [amount] WETH from Arbitrum One back to Ethereum and then trade it for USDC.",
            "Exchange all existing tokens in my wallet for ETH. Once, finished send it to [CEX deposit wallet]",
            "when gas is below 10, harvest the eth yield from Dolomite and deposit the eth to Rodeo", # "When gas is below 10, harvest the Eth yield from xyz and deposit the Eth to abc",
            "Set stop loss for ETH on arbitrum chain. Sell 0.1 ETH when price goes lower than 1500",
            "Set limit orders to buy ETH each time the price dips 10% from the current price, buy for 100 USDC each time.",
            "each monday, claim my vested token from the Dolomite protocol, and sell it for ETH at market price.", # "Each monday, claim my vested token from the XYZ protocol / or give smart contract address, and sell it for ETH at market price.",
            "each monday, claim my vested tokens, and stake them in Jones jUSDC pool", # "Each monday, claim my vested tokens, and stake them in xyz pool",
            "Bridge 500 usdc each to linea, zk sync and base when gas is below 10",
            "buy xyz if its market cap falls below 50mn over the next 7 days",
            "Bridge 1 ether to arb chain",
            "Bridge 1 ether to Arbitrum via Hop",
            "Bridge 3.2 eth to mainnet",
            "Send $5 to [insert wallet] every 3 hours",
            "Convert 1E to USDC",
            "Swap usdc to usdt when usdc/usdt is more than 1.01",
            "Transfer 100 Coins staked on protocol X [dAPP link] to protocol Y [dAPP link].",
            "Swap all ETH from this address [address] into USDT and sends these USDT to this address [address]",
            "Bridge all ETH at this address [address] from the arbitrum blockchain to the ether blockchain",
            "When $TOK is at 12k market cap, buy $20 worth of $TOK",
            "Claim and compound rewards from GND protocol",
            "Claim rewards from TIG staking and tigUSD staking and then add that tigUSD back to tigUSD position in tigristrade everyday if the tigUSD rewards balance is above 5",
            "Swap 100 of 0x... to y",
            "Bridge 100 of x on base to bnb and swap to y",
            "Send/transfer 1 eth on base and give me usdc on zksyncbridge 1 eth from base to zksync and swap to usdc",
            "transfer 10 USDC to address",
            "transfer all USDC to address",
            "deposit 100 USDC into Morpho",
            "deposit all ETH-USDC LP into Dodoex",
            "lock all LP into Dodoex",
            "bridge all USDC from Arbitrum to Optimism with Bungee",
            "long asset with 10x leverage",
            "Increase the x asset's long position by keeping the leverage the same and adding xyz amount of asset y as collateral",
            "compound my staking rewards",
            "Swap asset # x for y, at % max price impact",
            "swap 100 usdc on syncswap every hours for 3 days",
            "bridge 100usdc on stargate every hour for 10 days when gas is less than 10",
            "remove 50% lp from (app) on the 15 October and swap USDT for USDC on 16 October. Act only when gas is below 20 gwei",
            "on the 16 October bridge 0.5 ETH to zk sync using (app name) when gas is below 20 gwei",
            "buy 0.2ETH of x when x is 0.5 USD price",
            "swap 100 usdc.e to eth on llamaswap and long eth with 5x leverage on gmx when eth is $1550",
            "Borrow 1000 usdc.e when usdc.e is above $1.005 and swap to usdt",
            "every morning at 9:00 am, claim deepp rewards and use that to buy dlp",
            "When btc goes below 25k, market buy eth at the best available price with 5000 USDC",
            "I have an existing DOLA USDC position on Aurat. Can you help me harvest my yield every week and compound into the same position?",
            "vote 100% for BNB/THENA pool on thena",
            "Claim and sell STG rewards for usdc",
            "dump all positions to cash",
            "buy dai every 37 min for 2 days", #"buy X token every 37 min for 2 days",
            "deposit usdc and eth into camelot usdc/eth pool, stake the spnft into a nitro pool", # "open a position for usdc/eth on camelot, lock the spnft and deposit into a nitro pool",
            "sell everything from multiple chains into jusdc",
            "when Eth price hits $1500 buy 3 eth worth using my $usdc Balance",
            "swap my $bitcoin for usdc",
            "disperse 0.1E to these 10 wallets",
            "When gas is below 7, swap 0.01 ETH to USDC, repay 50% of USDC loan on Aave, withdraw 50% of supplied ETH on Aave.",
            "Buy 1 eth with USDT if price goes to 1200$",
            "Swap 100 of token A for token B",
            "swap for z token",
            "swap weth into eth", # "Unwrap Weth into Eth, find the fastest & cheapest dex or dex aggregator to do it in",
            "trade 1 baby bear for eth then swap that eth for usdc and send that usdc over to the arbitrum network with hop exchange",
            "Disperse ETH into few wallets",
            "every saturday at 18:00 claim incentives from my velodrome lock positions. sell all incentives into velo. lock that velo into the same lock position.", # "Every saturday at 18:00 claim incentives from my velodrome lock positions, if the fees are more than $100 worth claim them as well. Sell all incentives and fees claimed if any into VELO. Lock that VELO into the same lock position.",
            "If token X goes -30%, sell to USDC",
            "buy xxx$ of $TICKER with stablecoins/from my usdc",
            "bridge 20$ from polygon to arbitrum", # "bridge x$ from polygon to arbitrum",
            "Bridge ETH from mainnet to Arbitrum",
            "Swap ETH to USDC",
            "craft out the best route to get usdc from cosmos to avalanche",
            "cheapest route for eth to arbitrum now",
            "sell eth for usdc. Move usdc to wallet 2. Turn usdc into (some shitcoin)",
            "Sell eth to usdc. Move usdc to Coinbase wallet on Tuesday",
            "Stake Btrfly. Restake rewards on may 15th",
            "once a week, when gas is below 10, claim my wjAURA - WETH LP rewards on balancer. Deposit all AURA into Jones' wjAURA and stake BAL as auraBAL.",
            "Each day, claim my ARB rewards on Jones DAO from the jGLP & jUSDC masterchef contract and convert to $ETH",
            "bridge 1 ETH from mainnet to zksync and swap 0.5 ETH to USDC on zksync",
            "once eth balance hits X, buy Y of $token when gas is below z",
            "on every Wednesday at 10am until the end of november send 500usdt to address 0x-------- and $500 worth of Eth to 0x---- when gas is below 10",
        ]
        for u in usermsgs:
            perform_message(u.lower(), prompt, save)
    if step == 0 or step == 22:
        usermsgs = [
            "on GMX v2, when ETH price is near x, can you deposit x USDC collateral to avoid liquidation",
            "can you gather all the ETH that I have spread around different chains and bridge it in the most cost-efficient form to Arbitrum",
            "Bridge eth to zkynsc swap half of eth to usdc then stake it",
            "If Ldo goes down 10% or has a onchain liquidation bid it",
            "Sell [X] NFT to the highest bidder on blur or opensea, bridge ETH to arbitrum and send to wallet: 0x..... (my other wallet)",
            "Claim liveTHE rewards once they are worth greater than 10$ and compound them back into liveTHE",
            "Claims LiveTHE rewards once they are greater than 10$ and swap to USDT",
            "Vote for the highest yielding bribe on thena on Wednesday at 2355 UTC",
            "bridge 0.1 ETH from Base to Linea",
            "Swap $50 of ETH into USDC on Kyberswap",
            "Deposit $25 USDC in Satori and take a long position on ETH with a SL and TP of ... . Use the rest of the USDCs to reach at least $200 trading volume using USDC/USDT pair",
            "If gas costs are below 10 and ETH-LONG liquidity on GMX is over 95% utilized, deposit 1000 USDC at 15x leverage",
            "disperse 0.1E to these 10 wallets",
            "long asset with 10x leverage; Add TP at x price and add SL at y price",
            "Use asset x as collateral for this long",
            "Increase the x assets long position by keeping the leverage the same and adding xyz amount of asset y as collateral",
            "compound my staking rewards",
            "Add liquidity to asset xs LP pool in the y-z price range",
            "Swap asset x for y, at % max price impact",
            "Calculate the $ value of slippage for executing an  <insert size> swap for <token pair / liquidity pool>",
            "Execute Stop losses (with specific parameters defined ofc)",
            "When <whale> makes a trade, automatically copy trade",
            "Vote for <pool> on <protocol> every Wednesday at 2355H (UTC)",
            "Unvest 300000 Trove from Nitro Cartel and swap to USDC; Revest remaining Trove at 16:00 UTC",
            "Deposit USDC into Jones jUSDC and deposit to Rodeo Finance USDC: (contract address)",
            "bridge eth from mainnet to arb when gas is below 15",
            "create lp pair rdpx/eth and compund rewards every week.",
            "claim rewards from Thena/Velodrome/Aerodrome, convert them to ETH and bridge them to Ethereum",
            "Whenever I have over $4000 in Ethereum, send $3500 to the gemini address, but only do this once a week at most",
            "bridge 1 ETH from mainnet to zksync and swap 0.5 ETH to USDC on zksync",
            "Long/short X if specific influencer tweets a ticker",
            "When $TOK is at 12k market cap, buy $20 worth of $TOk",
            "Short BTC at 38K with 50x leverage, long at 38k if 'X' twitter account tweets bearish",
            "which ever token this wallet (provides the wallet to be tracked) buys with a minimum of 0.1eth, buy 0.1eth of it as well",
            "Claim and sell and STG rewards",
            "Sell all tokens on the eth network below 10$ in 1 transaction",
            "maintain healthy collateral ratio on synthetix. when gas is below 10, rebalance.",
            "collect weekly synthetix rewards, claim as soon as gas is below 10",
            "Bridge 500 usdc  each to linea, zk sync and base when gas is below 10",
            "Provide liquidity to the most liquid pair on linea, zk sync and base",
            "buy xyz if its market cap falls below 50mn over the next 7 days",
            "on every Wednesday at 10am until the end of november send 500usdt to address 0x-------- and $500 worth of Eth to 0x---- when gas is below 10",
            "leverage long using curveusd with 10 ETH, close it once health drops below 1.1",
            "harvest my position on sushi and also my position on camelot",
            "harvest on sushi and sell/restake.",
            "Sell eth for usdc. Move usdc to wallet 2. Turn usdc into (some shitcoin)",
            "Sell eth to usdc. Move usdc to Coinbase wallet on Tuesday",
            "Stake Btrfly. Restake rewards on may 15th (etc)",
            "carry out x no of swap on the dapp  daily for 1 month when gas is less than 30",
            "help me snipe token A on block zero, if gas fee is above 30$ dont take the trade",
            "Execute a perpetual basis spread trade focusing on funding rate arbitrage",
            "Identify and perform a spot DEX to on-chain perpetual arbitrage opportunity",
            "Utilize Opyns SQUEETH to hedge my position between ETH and USDC my position details currently are...",
            "Allocate collateral to Dopexs Atlantic ETH straddle, and acquire the most affordable hedge lasting for five Atlantic epochs (Note: Each epoch spans 2 days)",
            "Upon identifying a 2% price reduction (post-fees) for a Dopex $1600 ETH Call compared to a similar tenor, asset, and strike price option on Lyra, initiate a call parity trade by purchasing 10 Dopex calls and selling 10 corresponding calls on Lyra to capitalize on the spread/parity",
            "Employ spot and money markets like AAVE and Silo to execute pair trades, for instance, lending ETH, borrowing BTC, and selling the borrowed BTC for ETH.",
            "Leverage on-chain perpetuals like GMX and Gains Network for pair trading, while possibly utilizing spot or money markets like Aave to enter the most efficient short on SOL/DOT.",
            "Discover the optimal yield for my ETH, including scenarios where its used as margin to borrow another higher-yield asset",
            "Engage in cross-chain yield arbitrage, such as borrowing ETH on AAVE Mainnet at 2% and lending it on AAVE Polygon for 10%",
            "Given my trust in USDT, USDC, DAI, and preference for Mainnet, Arbitrum, and Polygon chains, provide yield options available on AAVE, Silo.finance, MakerDAO, and Compound",
            "Possessing ETH and desiring to short stETH against it, determine the most cost-effective approach to execute this on Mainnet (e.g., utilizing AAVE etc).",
            "Swap 100 of 0x... to y ",
            "Bridge 100 of x on base to bnb and swap to y",
            "Send/transfer 1 eth on base and give me usdc on zksync",
            "Ok i want to 3x leverage GLP and zap out of USDC into some other asset thats support on Dolomite",
            "Hey Spice Bot, if price of ETH goes up and Health Factor on AAVE loan goes below 1.5 , borrow more USDC and buy more ETH if price hasnt increased more than 5% in past 24h, and make sure gas is below N before you do it all",
            "Claim fSHROOMIEZ on Caviar as soon as reward > 1.1 and add back to Baton / Caviar LP with ETH counterpart, if/as soon as gas is below N; give me 10x leverage on the new crvUSD/sFRAX pools by looping my position accordingly",
            "Send one Hedgies NFT to [inputs address]",
            "craft out the best route to get usdc from cosmos to avalanche for example",
            "cheapest route for eth to arbitrum now",
            "give me the highest yield strategy on ETH now and execute for me",
            "Bridge all my eth from Arbitrum to ethereum",
            "Swap 200 usdt for btc when btc price is 26700$",
            "Sell x% of x$coin when price is x",
            "Disperse ETH into few wallets",
            "Find profitable Arb opportunities on chain",
            "when gas is under 20, bridge 1 eth to base and swap to $wig",
            "short Eth if it goes below or touches xxx price",
            "Stake 3eth at lido for 2 weeks",
            "Deposit $500 usdc and 0.35eth into an lp on uniswap v3",
            "deposit 0.35 eth into aave",
            "borrow $400 usdc and swap to $BITCOIN",
            "lock steth for 2months",
            "Move all assets in active wallet to *x-adress",
            "when eth is below $1600, buy $500 usd worth each week",
            "please monitor the APR on dexter, kujira blue and compare it to osmosis APR. Let me know when they offer better terms",
            "when gas is below 15, deposit 10 eth as collateral in Aave. Then take out a loan of 10,000 usdc against that eth. Then take that 10,000 usdc and deposit it into Rollbit",
            "deposit 2 eth as collateral into Rage trade on arbitrum. Once Eth price goes below $1,750, long Eth with 5x leverage",
            "buy 5,000 usdc worth of eth and then bridge it to zk sync era using orbiter bridge. Then at noon each day for the next week swap $500 of eth for usdc and then swap back $500 worth of usdc for eth",
            "Whenever $WINR (0xD77B108d4f6cefaa0Cae9506A934e825BEccA46E) falls below 3$, swap 2eth for it",
            "Whenever a Milady NFT (input ca) gets listed under 1.5eth, buy it",
            "if an NFT of this collection gets listed 20% below floor price buy it and instantly list it at floor",
            "whenever apr on this platform goes below APR on this other platform, liquidate first position to open it on this other platform",
            "When gas is below 7, swap 0.01 ETH to USDC, repay 50% of USDC loan on Aave, withdraw 50% of supplied ETH on Aave.",
            "when Eth price hits $1500 buy 3 eth worth using my $usdc Balance",
            "swap my $bitcoin on the dex with the least amount of slippage",
            "At exactly 19:05 UTC, bridge $50 ETH to Starknet ETH",
            "bridge eth to arb when gas is sub .5$ and swap back when arb hits .90",
            "transfer 10 USDC to address",
            "transfer all USDC to address",
            "deposit 100 USDC into Morpho",
            "deposit all ETH-USDC LP into Dodoex",
            "lock all LP into Dodoex",
            "bridge all USDC from Arbitrum to Optimism with Bungee",
            "sell all tokens with balances under $100 and convert to USD",
            "Sell all of my tokens under $100 and convert to USDT on mainnet",
            "Remove lent assets from Timeswap pool at maturity (and notify me)",
            "Bridge and swap 0.2 ETH into required asset and lend into new Timeswap pool as soon as one launches",
            "Unwind my borrow position on Dolomite when borrow APY rises above 69 %",
            "Sell my XYZ as soon as 0x123 sells",
            "Copytrade 0x123 with proportional position sizes if gas cost doesnt exceed 1/10 of position size",
            "Pull my liquidity on Uniswap if price falls below X",
            "Find the cheapest onchain put on ETH with strike price below 1234",
            "Compare and list onchain yield opportunities for NFT X or asset Y",
            "Sell half of asset X as soon as price doubles from my entry",
            "Set A trade for 9am(New York time) Long position on eth using a 1000$ margin and close trade when 100 dollars is lost or when 150 is gain",
            "On 30th OCT stack 100eth for 6months on Mantle ",
            "Sell X% of Y coin (or insert contract address?) every (day of week) at 12am utc on (insert preferred DEX).",
            "Transfers 100 Coins staked on protocol X [dAPP link] to protocol Y [dAPP link]",
            "Swaps all ETH from this address [address] into USDT and sends these USDT to this address [address]",
            "Transfers all ETH at this address [address] from the arbitrum blockchain to the ether blockchain.",
            "swap USDC for Arb if the price reaches $0.90 (ideally it would be smart enough to recognize that I mean once it DROPS to $0.90)",
            "When gas settles down consolidate my ETH to this address (would love it if its smart enough to detect relatively low gas vs having to put up an absolute number ",
            "If ETH hits 1200 USD open a 12x long on GMX",
            "send 0.01E to [wallet/contract address] every hour for 5 days",
            "send 0.2E 20 times to [wallet/contract address] today",
            "Bridge 1 ether to arb chain",
            "Bridge 1 ether to Arbitrum via Hop",
            "Bridge 3.2 ethereums to mainnet",
            "Send $1000 to [insert wallet] every",
            "Send $5 to [insert wallet] every 3 hours",
            "Convert 1E to stablecoin",
            "Swap usdc to usdt when price different is more than 1%",
        ]
        for u in usermsgs:
            perform_message(u.lower(), prompt, save)


# examples that dont work because we explicitly do not support them yet
def antisuite(prompt=1, save=False):
    # working just not integrated
    perform_message("get money from blur, send to arb, open up position on NFT perp all in one go", prompt, save)
    perform_message("I want to stake my arb, please give me 3 options", prompt, save)
    perform_message("What I do with my stETH?", prompt, save)
    
    # not working
    perform_message("the idea is youre minting DPXETH, you need USDC and RPDX and you need to bond and do a bunch of steps", prompt, save)
    perform_message("Leveraged looping on Aave", prompt, save)
    perform_message("pull my univ3 LP at price X and redeploy to range Y", prompt, save)
    perform_message("copy trade address X if token Y is greater than Z mcap with relative position size if it exceeds a certain threshold", prompt, save)
    perform_message("Borrow against my NFT on [Protocol X] at 60% LTV for 30 days, and stake the ETH on [Protocol Y]. 1 day before the loan is due, unstake and repay the loan", prompt, save)
    perform_message("compose transaction that claims rewards from existing positions (where rewards are >$2), convert and/or stake reward to appropriate vault, and update bribes based on current allocation weight", prompt, save)
    perform_message("sell all my CRV for Pendle on Arbitrum, lock it for 1 month, and when possible vote for the rETH pool", prompt, save)
    perform_message("What tokens are releasing on this chain", prompt, save)
    perform_message("when pool2 unlock threshold gets hit I want to remove my funds", prompt, save)
    perform_message("Uniswap LP management", prompt, save)
    perform_message("GLP farming. Like creating an algo to exit and enter at correct times", prompt, save)
    perform_message("Wants to loop with pendle to earn yield; have to go pendle to buy PT, then go to a market to borrow PT, then go back to pendle to buy PT; i want my LTV at 60%” - loop until it gets there", prompt, save)
    perform_message("Lending rate and borrowing rates for stable coins across different markets and chains are all different - thinks he would use this product to verify and quickly bridge, deposit, borrow, etc. to arb", prompt, save)
    perform_message("Update the range on my Uniswap position to the recommended values", prompt, save)
    perform_message("Monitoring uni v3 lp position and constantly balance it to keep it in a certain range or adjusting the range", prompt, save)
    perform_message("arbitrage bot: buy btc on x and sell on y until price equilizes", prompt, save)
    perform_message("If the pools total volume decreases by x% pull all of my funds in a frontrun", prompt, save)
    perform_message("help me snipe token A on block zero, if gas fee is above 30$ don't take the trade", prompt, save)  
    perform_message("sell all of my tokens under $100 and convert to usdt on mainnet", prompt, save)
    perform_message("Deposit 1000 USDC and borrow ARB on Paraspace, bridge from Arbitrum to Ethereum, deposit ARB and borrow USDC on Compound, deposit USDC on GMD Protocol", prompt, save) # perform_message("Go to ParaSpace on Arbitrum deposit collateral ($ETH or $USDC) apy 3.9% and 2.7%, Take a loan from $ARB apy 1%, Go to compound and deposit $ARB and borrow $USDC apy 2.3%, Go to @GMDprotocol and deposit $USDC apy 7%", prompt, save)
    usermsgs = [
        "make __(opensea, blur) bid at so so and so time",
        "Send 1000 USDC to 0x2B605... when wallet value falls below $2000",
        "Claim YT-swETH yield on Pendle when gas fees are less than 5% of reward",
        "Remove lent assets from Timeswap pool at maturity",
        "Bridge and swap 0.2 ETH into required asset and lend into new Timeswap pool as soon as one launches",
        "Sell my XYZ as soon as 0x123 sells",
        "Copytrade 0x123 with proportional position sizes if gas cost doesn’t exceed 1/10 of position size",
        "Find the cheapest onchain put on ETH with strike price below 1234",
        "Compare and list onchain yield opportunities for NFT X or asset Y",
        "Set A trade for 9am(New York time) Long position on eth using a 1000$ margin and close trade when 100 dollars is lost or when 150 is gain",
        "On 30th OCT stack 100eth for 6months on Mantle",
        "If liquidity withdrawn in pool X is above 10M, withdraw asset A",
        "Deposit $500 usdc and 0.35eth into an lp on uniswap v3",
        "Recommend me some pools to farm yield with above 10% APY using my ETH. Parameters include - farm TVL >$5mil, rewards are majority in ETH and not native tokens.",
        "Send one Hedgies NFT to [inputs address]",
        "Provide liquidity to the most liquid pair on linea, zk sync and base",
        "Sell [X] NFT to the highest bidder on blur or opensea, bridge ETH to arbitrum and send to wallet: 0x",
        "At Wednesday 11 PM UTC, vote for the pool with highest rewards in Chronos with APR greater than 150%",
        "Add liquidity to asset x's LP pool in the y-z price range",
        "alert when (account address) buys for more than 1ETH",
        "copytrade (account address) with 0.5 buy if he buys for  more than 1ETH",
        "Hey Spice Bot, if price of ETH goes up and Health Factor on AAVE loan goes below 1.5 , borrow more USDC and buy more ETH if price hasn't increased more than 5% in past 24h, and make sure gas is below N before you do it all",
        "Claim fSHROOMIEZ on Caviar as soon as reward > 1.1 and add back to Baton / Caviar LP with ETH counterpart, if/as soon as gas is below N",
        "give me 10x leverage on the new crvUSD/sFRAX pools by looping my position accordingly",
        "if gas costs are below 10 and ETH-LONG liquidity on GMX is over 95% utilized, deposit 1000 USDC at 15x leverage",
        "which ever token this wallet (provides the wallet to be tracked) buys with a minimum of 0.1eth, buy 0.1eth of it as well",
        "Give me the 3 best DAI farms on ethereum",
        "Tell me the cheapest way to to open a position on X based on what I have in my wallet",
        "What is the cheapest way for me to leverage up to go long on ETH and CRV",
        "I want to farm USDC/DAI LP. Where do I get the best yields for this liquidity position and how do I enter the position",
        "make me a scanning of these addresses: XXX, XXX, and make a speadsheet with my different positions (name, money in $, time since I'm holding position, PNL on position, etc)",
        "check my positions on velodrome and aerodrome and 1 hour before the end of epoch tell me which pool should I vote for giving me the best potential APR",
        "when this address XXX has less than 0.05E send it 0.2E",
        "give me the best performing addresses that had the biggest combined PNL in % and the biggest combined PNL in $ of these token contracts XXX XXX",
        "give me an alert if $BNB break the 200$ support, give me an alert whenever stablecoin inflow see a significant growth over 30D period (atleast +10%)",
        "check defillama and give me everyweek a review of which protocols and which chains are gaining the biggest traction (in TVL inflow, DAU)",
        "when a buy greater than X size takes place on tokenX sell %x of my position",
        "list all my nfts for sale on blur",
        "dump all my nfts",
        "anytime this wallet buys, duplicate trade at x% of portfolio",
        "these 6 wallets on your monitoring list purchased XXX in amounts greater than XXX, I have constructed a trade that sells XX% of your ETH to purchase XXX of token XXX this is XX% of your portfolio / wallet balance.  Click here to execute",
        "claim my $tia airdrop for any wallet that is eligible in this set: addr1, addr2",
        "If by date_1 farming rewards > 99$ then claim, else compound",
        "Anytime between date_1 and date_2, if  funding rate >1% and ETH > 1600, short ETH using $1000 USDC",
        "Every 1st and 15th of the month revoke all spending permissions on wallet1 where limit > 1.",
        "approve x token on uniswap router",
        "Find profitable Arb opportunities on chain",
        "give me the highest yield strategy on ETH now and execute for me",
        "Provide me a snapshot of my current portfolio across x,y,z wallets and all chains with a token balance. Show me the % changes in weighting and value over the last month.",
        "Calculate the $ value of slippage for executing an  <insert size> swap for <token pair / liquidity pool>",
        "Execute Stop losses (with specific parameters defined ofc)",
        "When <whale> makes a trade, automatically copy trade",
        "collateralize LST on polygon zkEVM, borrow DUSD, deposit DUSD in DUSD/USDC farms",
        "withdraw all my funds from the SP-BLUR Spice FInance vault, when the vault makes Blur tokens claimable, claim the tokens and sell them for ETH",
        "When gas is below $20 on ETH Mainnet buy me 0.05ETH, bridge 0.03ETH under arb layer 2 and leave 0.01ETH in my mainnet account",
        "when token x equals price y or less buy in increments of 0.5 eth no closer than every 15 minutes up to 5 eth using the cheapest aggregator (or maybe specify say 1inch) and then send to wallet b",
        "Buy 0.35 eth worth of $MOG and sell at a 2x",
        "Track this wallet, spend $100 on the coin whenever this wallet makes a purchase >$100k on a coin",
        "do a swap 10 usdt to zkusd on syncswap on all my accounts, at randomized time(random range is 8 hours)",
        "When lock period expires on date_1 unbind LP and sell the lower of the two balances for the other staking resulting balance in vault",
        "if btc price is above 27000 on a 4 hour close add 20% size to gmx v2 long",
        "when ltv is greater than 80%, repay 10% of loan.", # "When LTV is greater than 80%, repay loan or add collateral x amount of collateral so that LTV is < 70.",
        "buy with more usdc as the price goes lower, eg increase the buy usdc amount by 20% each 10% price drop",
        "leverage long using curveusd with 10 ETH, close it once health drops below 1.1",
    ]

# if __name__ == "__main__":
    # if len(sys.argv) > 2:
        # i = int(sys.argv[1])
        # j = int(sys.argv[2])
        # suite(step=i, prompt=1, save=j)
    # else:
        # print("pass integers representing step and savefile as command line parameters")
perform_message("bridge 0.02 eth to arbitrum")