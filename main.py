import asyncio
import inspect
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Any, cast

import httpx
import numpy as np
import openai
import pandas as pd
import zoneinfo
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import Json
from scipy import spatial
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import Field, Relationship, SQLModel, desc, select
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

load_dotenv()
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
testai = os.getenv("TESTAI")
entities_url = (
    "http://127.0.0.1:5000/verified-entities?simple=true"
    if testai == "true"
    else "http://127.0.0.1:5000/verified-entities?simple=true"
)
app = FastAPI(
    middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])]
)
timeout = 55
engine = create_async_engine(os.getenv("DB_URL") or "?", echo=False)
Session = async_sessionmaker(engine)


async def chat(**kwargs):
    return await research(kwargs["query"])


async def support(**kwargs) -> Any:
    resp = await make_request_with_retries(entities_url)
    if not resp:
        raise Exception("get verified entities failed.")
    newps = []
    cn = chainidstonames()
    for ps in resp["protocols"]:
        toadd = {}
        toadd["name"] = ps["name"]
        toadd["pools"] = ps["pools"]
        newc = []
        for cs in ps["chains"]:
            newc.append(cn[cs])
        toadd["chains"] = newc
        prs = []
        for k, v in list(resp["actions"].items()):
            if ps["name"].lower() in v:
                prs.append(k)
        toadd["actions"] = list(set(prs))
        newps.append(toadd)
    return json.dumps(
        to_lowercase(
            {
                "protocols": newps,
                "actions": resp["actions"],
                "chains": [c["name"] for c in resp["chains"]],
                "tokens": ["all", "any"],
            }
        )
    )


# {
# "name": "positions",
# "description": "Get user's positions and token balances",
# "parameters": {
# "type": "object",
# "properties": {
# "userAddress": {
# "type": "string",
# "description": "Blockchain address of the user"
# }
# },
# "required": [
# "userAddress"
# ],
# "unevaluatedProperties": false
# }
# },
async def positions(**kwargs) -> Any:
    url = f"https://pro-openapi.debank.com/v1/user/all_token_list?id={kwargs['userAddress']}"
    headers = {"AccessKey": os.getenv("DEBANK_ACCESS_KEY") or "?"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
    try:
        response.raise_for_status()
    except Exception as e:
        print(e)
        return []
    data = response.json()
    newdata = []
    for d in data:
        if "is_verified" not in d or not d["is_verified"]:
            continue
        datadeletekeys = [
            "id",
            "decimals",
            "logo_url",
            "is_core",
            "is_wallet",
            "time_at",
            "raw_amount",
            "raw_amount_hex_str",
            "is_verified",
        ]
        for ddk in datadeletekeys:
            d.pop(ddk, None)
        if "optimized_symbol" in d:
            d["symbol"] = d["optimized_symbol"]
        elif "display_symbol" in d:
            d["symbol"] = d["display_symbol"]
        d.pop("optimized_symbol", None)
        d.pop("display_symbol", None)
        if "price" in d and (not d["price"] or d["price"] == 0):
            d.pop("price", None)
        if "price_24h_change" not in d or not d["price_24h_change"]:
            d.pop("price_24h_change", None)
        sc = get_shortened_chains()
        if "chain" in d and d["chain"] in sc:
            d["chain"] = sc[d["chain"]]
        newdata.append(d)
    return json.dumps(newdata)


async def protocols_positions(**kwargs) -> Any:
    url = f"https://pro-openapi.debank.com/v1/user/all_complex_protocol_list?id={kwargs['userAddress']}"
    headers = {"AccessKey": os.getenv("DEBANK_ACCESS_KEY") or "?"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    newdata = []
    for d in data:
        datadeletekeys = [
            "id" if d["name"] else "name",
            "chain",
            "logo_url",
            "has_supported_portfolio",
            "tvl",
        ]
        for ddk in datadeletekeys:
            d.pop(ddk, None)
        newp = []
        for p in d["portfolio_item_list"]:
            itemdeletekeys = ["update_at", "position_index"]
            for idk in itemdeletekeys:
                d.pop(idk, None)
            newp.append(p)
        newdata.append(d)
    return json.dumps(newdata)


def transfer(**kwargs):
    return "/transfer has been called successfully"


def swap(**kwargs):
    if "outputToken" in kwargs:
        return json.dumps(
            {"outputAmount": "outputAmount", "outputToken": kwargs["outputToken"]}
        )
    return json.dumps({"outputAmount": "outputAmount"})


def bridge(**kwargs):
    if "token" in kwargs:
        return json.dumps(
            {"outputAmount": "outputAmount", "outputToken": kwargs["token"]}
        )
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})


def deposit(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})


def withdraw(**kwargs):
    if "token" in kwargs:
        return json.dumps(
            {"outputAmount": "outputAmount", "outputToken": kwargs["token"]}
        )
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})


def claim(**kwargs):
    return json.dumps({"outputAmount": "outputAmount", "outputToken": "outputToken"})


def borrow(**kwargs):
    if "token" in kwargs:
        return json.dumps(
            {"outputAmount": "outputAmount", "outputToken": kwargs["token"]}
        )
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


def get_available_functions():
    return {
        # "positions": positions,
        "support": support,
        # "chat": chat,
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
        "time": tim,
    }


def chainnamestoids():
    return {
        "arbitrum": 42161,
        "arbitrum one": 42161,
        "avalanche": 43114,
        "base": 8453,
        "binance": 56,
        "binancesmartchain": 56,
        "blast": 81457,
        "bnb smart chain": 56,
        "bnb": 56,
        "bsc": 56,
        "canto": 7700,
        "celo": 42220,
        "classic": 61,
        "cronos": 25,
        "ethclassic": 61,
        "ethereum": 1,
        "fantom": 250,
        "filecoin": 314,
        "filecoin-mainnet": 314,
        "gnosis": 100,
        "homestead": 1,
        "kava": 2222,
        "linea": 59144,
        "linea-mainnet": 59144,
        "mainnet": 1,
        "mantle": 5000,
        "matic": 137,
        "mode": 34443,
        "moonbeam": 1284,
        "moonriver": 1285,
        "optimism": 10,
        "op mainnet": 10,
        "polygon": 137,
        "zksync": 324,
        "zksync era": 324,
    }


def chainidstonames():
    return {
        "42161": "arbitrum",
        "43114": "avalanche",
        "8453": "base",
        "56": "binance",
        "81457": "blast",
        "7700": "canto",
        "42220": "celo",
        "61": "ethclassic",
        "25": "cronos",
        "1": "ethereum",
        "250": "fantom",
        "314": "filecoin",
        "100": "gnosis",
        "2222": "kava",
        "59144": "linea",
        "5000": "mantle",
        "137": "matic",
        "1284": "moonbeam",
        "1285": "moonriver",
        "10": "optimism",
        "324": "zksync",
        "34443": "mode",
    }


def get_protocols(production=False):
    if not production:
        return {
            "uniswap",
            "sushiswap",
            "pancakeswap",
            "kyberswap",
            "gmx",
            "aave",
            "kwenta",
            "compound",
            "radiant",
            "morpho",
            "rocket pool",
            "jonesdao",
            "rodeo",
            "kyber",
            "mmf",
            "camelot",
            "curve",
            "thena",
            "plutus",
            "trader joe",
            "solidly",
            "redacted cartel",
            "blur",
            "balancer",
            "yearn",
            "chronos",
            "dolomite",
            "lodestar",
            "stargate",
            "pendle",
            "sushi",
            "velodrome",
            "gmd",
            "dodo finance",
            "paraspace",
            "hop",
            "redacted",
            "spice",
            "polylend",
            "wing-finance",
            "paraspace-lending-v1",
            "woofi",
            "symbiosis finance",
            "mav",
            "spice finance",
            "nitro cartel",
            "rodeo finance",
            "defillama",
            "lido",
            "timeswap",
            "gnd",
            "tigristrade",
            "dodoex",
            "bungee",
            "syncswap",
            "llamaswap",
            "deepp",
            "aura",
            "coinbase",
            "binance",
            "jones dao",
            "jones",
            "rocketpool",
            "synthetix",
            "rollbit",
            "orbiter",
            "dapp",
            "gemini",
            "rage trade",
            "ragetrade",
            "silo",
            "ambient finance",
            "ambient",
            "velodrome",
            "aerodrome",
            "swell",
            "trader joe",
        }  # global protocols
    else:
        return {"gmx"}  # return protocols that are also tokens here


def get_chains(production=False):
    return {
        "mainnet",
        "ethereum",
        "base",
        "arbitrum",
        "arbitrum one",
        "optimism",
        "avalanche",
        "zksync",
        "bsc",
        "blast",
        "binancesmartchain",
        "binance",
        "polygon",
        "matic",
        "gnosis",
        "fantom",
        "canto",
        "mantle",
        "manta",
        "zora",
        "linea",
        "starknet",
        "cosmos",
        "metis",
        "scroll",
        "sepolia",
        "zksync era",
        "mode",
        "zircuit",
        "bnb smart chain",
        "smart chain",
        "binance smart chain",
        "bnbsmartchain",
        "bnb",
        "op mainnet",
        "linea-mainnet",
        "homestead",
    }  # global chains


def get_pools(production=False):
    if not production:
        return {
            "eth-usdc",
            "tricrypto",
            "weth-grail",
            "eth-jones",
            "pt-glp",
            "usdc-usdt",
            "wjaura",
            "jaura",
            "livethe",
        }
    else:
        return set()


def get_tokens(keywords=None):
    if not keywords:
        keywords = get_keywords()
    tokens0 = [
        "eth",
        "usdc",
        "usdt",
        "dai",
        "weth",
        "btc",
        "wbtc",
        "op",
        "ohm",
        "grail",
        "glp",
        "uni",
        "pepe",
        "pls",
        "stg",
        "arb",
        "jones",
        "plsjones",
        "lp",
        "pt",
        "smp",
        "lmp",
        "xgrail",
        "bitcoin",
        "btrfly",
        "dpx",
        "rdpx",
        "plsdpx",
        "coin",
        "sweed",
        "dodo",
        "saint",
        "vethe",
        "rlbtrfly",
        "livethe",
        "tigusd",
        "usdc.e",
        "usdce",
        "bal",
        "spnft",
        "wig",
        "gwei",
        "merit circle",
        "fxs",
        "usdcb",
        "sol",
        "wsol",
        "espls",
    ]  # global tokens
    with open("coins.json", "r", encoding="utf-8", errors="ignore") as f:       
        q = json.load(f)
        tokens0.extend(
            [
                x["symbol"].lower()
                for x in q
                if x["symbol"] not in keywords
                and x["symbol"] != ""
                and not any(
                    z in x["symbol"]
                    for z in [
                        ".",
                        "+",
                        "*",
                        "?",
                        "^",
                        "$",
                        "(",
                        ")",
                        "[",
                        "]",
                        "{",
                        "}",
                        "|",
                        "\\",
                    ]
                )
                and not all(zz.isdigit() for zz in x["symbol"])
            ]
        )
    return tokens0


def get_keywords(actions=None):
    if not actions:
        actions = list(get_available_functions().keys())
    keywords0 = [
        "condition",
        "time",
        "token",
        "protocol",
        "chain",
        "pool",
        "apy",
        "ltv",
        "fdv",
        "market cap",
        "health factor",
        "gas",
        "price",
        "loan",
        "outputtokens",
        "outputamount",
        "outputtoken",
        "redeposit",
        "amount_units",
        "start_time",
    ]
    keywords0.extend(actions)
    with open("words", "r") as file:
        words = file.read().split("\n")
    keywords0.extend([word.strip().lower() for word in words if word.strip()])
    keywords0.extend([f"{word.strip().lower()}s" for word in words if word.strip()])
    keywords0.extend([f"{word.strip().lower()}es" for word in words if word.strip()])
    keywords0.extend([f"{word.strip().lower()}d" for word in words if word.strip()])
    keywords0.extend([f"{word.strip().lower()}ed" for word in words if word.strip()])
    keywords0.extend([f"{word.strip().lower()}ing" for word in words if word.strip()])
    keywords0.extend(timezones())
    keywords0.extend(["ust"])
    keywords = set(keywords0)
    return keywords


def get_shortened_chains():
    shortened_chains = {
        "arb": "arbitrum",
        "eth": "ethereum",
        "op": "optimism",
        "avax": "avalanche",
        "ftm": "fantom",
        "bnb": "bsc",
        "bsc": "bsc",
    }
    return shortened_chains


def timezones():
    summer = datetime.fromisoformat("2023-06-21")
    winter = datetime.fromisoformat("2023-12-21")
    ans = set()
    for tz in zoneinfo.available_timezones():
        tz0 = zoneinfo.ZoneInfo(tz).tzname(summer)
        tz1 = zoneinfo.ZoneInfo(tz).tzname(winter)
        if tz0[0] not in ["+", "-"] and "/" not in tz0:
            ans.add(tz0.lower())
        if tz1[0] not in ["+", "-"] and "/" not in tz1:
            ans.add(tz1.lower())
    return list(ans)


def to_lowercase(data):
    if isinstance(data, str):
        if data == "outputAmount" or data == "outputToken":
            return data
        return data.lower()
    elif isinstance(data, list):
        return [to_lowercase(item) for item in data]
    elif isinstance(data, dict):
        return {key: to_lowercase(value) for key, value in data.items()}
    else:
        return data


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(
                    1
                    + min((distances[index1], distances[index1 + 1], new_distances[-1]))
                )
        distances = new_distances
    return distances[-1]


async def query_openai_for_multiple_replacements(message, multiple_replacements):
    prompt = "you are a crypto assistant. Given the context: '{}', choose the best replacement for each word from its options. simply return the corrected message and nothing else:\n".format(
        message
    )
    for word, replacements in multiple_replacements.items():
        prompt += "- For '{}', options are: {}. \n".format(
            word, ", ".join(replacements)
        )

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        corrected_message = response.choices[0].message.content.strip()

        # Check word count in the original and corrected messages
        original_word_count = len(message.split())
        corrected_word_count = len(corrected_message.split())

        # Return the original message if word counts differ
        if original_word_count != corrected_word_count:
            print(
                "Word count mismatch between original and corrected messages. Returning the original message."
            )
            return message

        return corrected_message
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return message


class Tracking(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_address: str
    inputted_query: str
    generated_api_calls: Json | None = Field(default=None, sa_type=JSONB)
    edited_api_calls: Json | None = Field(default=None, sa_type=JSONB)
    generated_transactions: Json | None = Field(default=None, sa_type=JSONB)
    first_simulation_status: int | None = Field(default=None)
    second_simulation_status: int | None = Field(default=None)
    executed_status: int | None = Field(default=None)
    created: int | None = Field(default=None)
    updated: int
    dataset: list["Dataset"] = Relationship(back_populates="tracking")

    class Config:
        arbitrary_types_allowed = True


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    query_id: int | None = Field(default=None, foreign_key="tracking.id")
    generated_correct: bool | None = Field(default=None)
    edited_correct: bool | None = Field(default=None)
    created: int | None = Field(default=None)
    updated: int | None = Field(default=None)
    tracking: Tracking | None = Relationship(back_populates="dataset")


async def check_cache(message) -> list[dict[str, Any]]:
    last_update = 1713820585
    async with Session.begin() as session:
        res = await session.execute(
            select(Tracking.generated_api_calls)
            .where(Tracking.inputted_query == message)
            .where(Tracking.updated > last_update)
            .order_by(desc(Tracking.updated))
            .limit(1)
        )
        rec = res.first()
        return rec.generated_api_calls if rec else []


async def track_query(address, message, test=True) -> Any:
    now = int(time.time())
    if not test:
        async with Session() as session:
            tracking = Tracking(
                user_address=address, inputted_query=message, created=now, updated=now
            )
            session.add(tracking)
            await session.commit()
            await session.refresh(tracking)
            ds = Dataset(query_id=tracking.id, created=now, updated=now)
            session.add(ds)
            await session.commit()
            await session.refresh(tracking)
        return tracking.id
    else:
        print(address, message)
        return -1


async def track_response(r, answer: list[dict[str, Any]], test=True) -> Any:
    if not test:
        async with Session.begin() as session:
            rec = await session.get(Tracking, r)
            if rec:
                rec.generated_api_calls = answer
    else:
        print(answer)


__cache: dict[str, dict[str, Any]] = {}


async def make_request_with_retries(url, max_retries=3, initial_delay=1) -> Any:
    if (
        url in __cache.keys()
        and datetime.now().timestamp() - __cache[url]["ts"] < 3 * 60 * 60
    ):
        return __cache[url]["data"]
    for retry in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)

            if (
                response.status_code == 200
                and response.json().get("status") == "success"
            ):
                data = response.json()  # Parse the response as JSON
                __cache[url] = {"data": data, "ts": datetime.now().timestamp()}
                return data  # Return the parsed JSON response
            else:
                print(f"Request failed with status code {response.status_code}")
                print("Response content:")
                print(response.text)  # Print the response content
        except Exception as e:
            print(f"Request failed with exception: {e}")

        if retry < max_retries:
            # Calculate the exponential backoff delay
            delay = initial_delay * (2**retry)
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print("Max retries reached. Request failed.")

    return None  # Return None if all retries fail


def remove_descriptions(obj):
    if isinstance(obj, dict):
        # For dictionary, create a new dictionary without 'description' keys
        return {k: remove_descriptions(v) for k, v in obj.items() if k != "description"}
    elif isinstance(obj, list):
        # For list, apply the function to each item in the list
        return [remove_descriptions(item) for item in obj]
    else:
        # For other types, return the object as is
        return obj


async def run_conversation_retry(
    message: str,
    prompt,
    user_address: str,
    available_functions: dict,
    startidx=0,
    support_chat=False,
):
    models = [
        "ft:gpt-3.5-turbo-0125:spice-finance::9WXQwsum",
        "ft:gpt-3.5-turbo-0125:spice-finance::9RNGfMaY",
        "ft:gpt-3.5-turbo-0125:spice-finance::9RMRhCWD",
        "ft:gpt-3.5-turbo-0125:spice-finance::9R8bEPIX",
        "ft:gpt-3.5-turbo-0125:spice-finance::9QyAX97O",
        "ft:gpt-3.5-turbo-0125:spice-finance::9HBHQMdV",
        "ft:gpt-3.5-turbo-0125:spice-finance::96KhUfB0",
        "ft:gpt-3.5-turbo-0125:spice-finance::96Iwe8Hz",
        "ft:gpt-3.5-turbo-0125:spice-finance::94dli8lA",
        "ft:gpt-3.5-turbo-0125:spice-finance::94cAMZFV",
        "ft:gpt-3.5-turbo-0125:spice-finance::94a1Z3hD",
        "ft:gpt-3.5-turbo-0125:spice-finance::94Ytq4jZ",
        "ft:gpt-3.5-turbo-0125:spice-finance::94Hzk2kt",
        "ft:gpt-3.5-turbo-1106:spice-finance::8bwuLclK",
        "gpt-3.5-turbo-0125",
        "gpt-4-turbo-preview",
        "gpt-4o",
    ][startidx:]
    results = []
    response = "Temporary AI Error, please try again"
    respidx = startidx
    for m in models:
        print("using", m)
        try:
            results, response = await run_conversation(
                message, prompt, user_address, available_functions, m, support_chat
            )
        except Exception as e:
            print("run convo fail", e)
            time.sleep(1)
            respidx += 1
            continue
        if response == "OpenAI Error":
            print(m, "OpenAI Error")
            time.sleep(1)
            respidx += 1
            continue
        return results, response, respidx
    return results, response, -1


async def run_conversation(
    message: str,
    prompt,
    user_address: str,
    available_functions: dict,
    modelStr: str,
    support_chat: bool,
):
    results = []
    behavior0 = """
    Solve the user's intent by executing a series of steps.
    Ensure steps are scheduled and triggered when necessary.
    Ensure all parts of the user's intent are solved.
    "all" is a valid input amount.
    Only use the functions you have been provided with.
    Only use the function inputs you have been provided with.
    Respond with a single sentence.
    Every function call should be unique.
    Never call multiple functions in parallel.
    """
    if support_chat:
        behavior0 = """
    Use the support endpoint to get the entities supported by our protocol, and respond to the user with complete and accurate information based on the returned data.
    Ensure your response is accurate based on the given data.
    For example, if a user asks where to short on base chain, and there are no protocols that support the short action and base chain, respond that there is nowhere to short on base chain.
    For example, if a user asks which protocols are supported on arbitrum, your answer should only contain protocols that support arbitrum.
    """
    # behavior0 = """
    # Use the provided functions to solve the user's intent via actions, conditionals, or scheduling.
    # If actions need to be scheduled or triggered, ensure that separate calls are made to do so.
    # Ensure all parts of the user's intent are solved.
    # Only use the function inputs you have been provided with.
    # Respond with a single sentence.
    # Never call multiple functions in parallel.
    # """
    behavior: list[openai.types.chat.ChatCompletionMessageParam] = [
        {"role": "system", "content": behavior0},
    ]
    query: list[openai.types.chat.ChatCompletionMessageParam] = [
        {"role": "user", "content": message}
    ]
    messages = behavior + query
    with open("tools-min.json", "r") as file:
        tools = json.load(file)
    # tools = remove_descriptions(tools)
    tools = [x for x in tools if x["function"]["name"] in available_functions]
    tokens = 0
    errors = 0
    # print(messages)
    while True:
        try:
            response = await client.chat.completions.create(
                model=modelStr,
                temperature=0,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                timeout=10 if not support_chat else 30,
                seed=14,
            )
            break
        except Exception as e:
            errors += 1
            time.sleep(2 * errors)
            if errors >= 2:
                return [], "OpenAI Error"
            else:
                print(e)
    tokens += response.usage.prompt_tokens if response.usage else 0
    # print(tokens)
    # print(response.system_fingerprint)
    response_message = response.choices[0].message
    innerbreak = False
    double_prev_response_message = None
    prev_response_message = None
    while response_message.tool_calls and not response_message.content:
        print(response_message)
        if prev_response_message and double_prev_response_message:
            tmpa = cast(
                openai.types.chat.ChatCompletionToolMessageParam,
                {
                    "name": response_message.tool_calls[0].function.name,
                    "arguments": json.loads(
                        response_message.tool_calls[0].function.arguments
                    ),
                },
            )
            tmpb = cast(
                openai.types.chat.ChatCompletionToolMessageParam,
                {
                    "name": prev_response_message.tool_calls[0].function.name,
                    "arguments": json.loads(
                        prev_response_message.tool_calls[0].function.arguments
                    ),
                },
            )
            tmpc = cast(
                openai.types.chat.ChatCompletionToolMessageParam,
                {
                    "name": double_prev_response_message.tool_calls[0].function.name,
                    "arguments": json.loads(
                        double_prev_response_message.tool_calls[0].function.arguments
                    ),
                },
            )
            if (
                (tmpa == tmpb or tmpa == tmpc)
                and "each" not in message
                and "every" not in message
            ):
                print("REPEATS")
                print("a", tmpa, "b", tmpb, "c", tmpc)
                break
        messages.append(
            cast(
                openai.types.chat.ChatCompletionAssistantMessageParam,
                {
                    "role": response_message.role,
                    "tool_calls": [c.model_dump() for c in response_message.tool_calls],
                },
            )
        )
        for ij, tool_call in enumerate(response_message.tool_calls):
            if tool_call.function.name == "buy":
                tool_call.function.name = "swap"
            if tool_call.function.name == "sell":
                tool_call.function.name = "swap"
            function_name = tool_call.function.name
            if (
                function_name in ["chat", "support", "positions"]
                and "spice-finance" in modelStr
            ):
                return [], "OpenAI Error"
            if tool_call.function.name == "multi_tool_use":
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    if (
                        "tool_uses" in function_args
                        and len(function_args["tool_uses"]) == 1
                    ):
                        tool_call.function.name = function_args["tool_uses"][0][
                            "recipient_name"
                        ].split(".")[-1]
                        function_name = tool_call.function.name
                        tool_call.function.arguments = json.dumps(
                            function_args["tool_uses"][0]["parameters"]
                        )
                except Exception as _e:
                    print(response_message, _e)
                    return [], "OpenAI Error"
            if function_name not in available_functions:
                innerbreak = True
                break
            function_to_call = available_functions[function_name]
            tool_call.function.arguments = re.sub(
                r"\[outputAmount\]",
                r'["outputAmount"]',
                tool_call.function.arguments,
            )
            tool_call.function.arguments = re.sub(
                r"\[outputToken\]",
                r'["outputToken"]',
                tool_call.function.arguments,
            )
            try:
                function_args = json.loads(tool_call.function.arguments)
            except Exception as _e:
                print(response_message)
                return [], "OpenAI Error"
            if isinstance(function_args, str) and function_name == "chat":
                innerbreak = True
                break
            if "slippage" not in message and "slippage" in function_args:
                del function_args["slippage"]
                tool_call.function.arguments = json.dumps(function_args)
            if "userAddress" in function_args:
                function_args["userAddress"] = user_address
            if ij != 0:
                prev = json.loads(
                    response_message.tool_calls[ij - 1].function.arguments
                )
                x = (
                    prev["outputToken"]
                    if "outputToken" in prev
                    else prev["token"]
                    if "token" in prev
                    else None
                )
                if (
                    x
                    and (
                        ("amount" in function_args and function_args["amount"] == "all")
                        or (
                            "inputAmount" in function_args
                            and function_args["inputAmount"] == "all"
                        )
                    )
                    and (
                        ("token" in function_args and function_args["token"] == x)
                        or (
                            "inputToken" in function_args
                            and function_args["inputToken"] == x
                        )
                    )
                ):
                    if "amount" in function_args:
                        function_args["amount"] = "outputAmount"
                    elif "inputAmount" in function_args:
                        function_args["inputAmount"] = "outputAmount"
                tool_call.function.arguments = json.dumps(function_args)
            results.append(tool_call.function)
            try:
                function_response = (
                    await function_to_call(**function_args)
                    if inspect.iscoroutinefunction(function_to_call)
                    else function_to_call(**function_args)
                )
            except Exception as e:
                print("function call error:", e)
                function_response = ""
            messages.append(
                cast(
                    openai.types.chat.ChatCompletionToolMessageParam,
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    },
                )
            )
        if innerbreak:
            break
        errors = 0
        # print(messages)
        while True:
            try:
                response = await client.chat.completions.create(
                    model=modelStr,
                    temperature=0,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    timeout=10 if not support_chat else 30,
                    seed=14,
                )
                break
            except Exception as e:
                errors += 1
                time.sleep(2 * errors)
                if errors >= 2:
                    return [], "OpenAI Error"
                else:
                    print(e)
        tokens += response.usage.prompt_tokens if response.usage else 0
        # print(tokens)
        # if tokens > 4097:
        # print(f'{tokens} TOKENS DETECTED')
        # print(response.system_fingerprint)
        double_prev_response_message = prev_response_message
        prev_response_message = response_message
        response_message = response.choices[0].message
        if len(messages) >= 20:
            break
    messages.append(
        cast(
            openai.types.chat.ChatCompletionAssistantMessageParam,
            {
                "role": response_message.role,
                "content": response_message.content,
            },
        )
    )
    return results, response_message


async def process_message_ai(
    message: str,
    user_address="",
    prompt=1,
    save=-1,
    skip_db=False,
    production=False,
    default_chain="",
) -> Any:
    if not isinstance(user_address, str):
        return {
            "calls": [],
            "message": "Non string user address",
            "message_id": -1,
        }
    if not isinstance(prompt, int):
        return {
            "calls": [],
            "message": "Non integer prompt",
            "message_id": -1,
        }
    if not isinstance(save, int):
        return {
            "calls": [],
            "message": "Non integer savefile",
            "message_id": -1,
        }
    print("user: ", user_address, message)
    if default_chain != "":
        try:
            default_chain = chainidstonames()[
                str(
                    chainnamestoids()[
                        default_chain.replace("chain", "").strip().lower()
                    ]
                )
            ]
        except Exception as e:
            print(1, e)
            try:
                default_chain = chainidstonames()[
                    default_chain.replace("chain", "").strip().lower()
                ]
            except Exception as e:
                print(2, e)
                default_chain = "ethereum"
    if message == "":
        return {"calls": [], "message": "", "message_id": -1}
    if message == "hello":
        return {"calls": [], "message": "hello", "message_id": -1}
    if message == "gm":
        return {"calls": [], "message": "gm", "message_id": -1}
    if message == "good morning":
        return {"calls": [], "message": "good morning", "message_id": -1}
    message = message.replace("\n", "")
    using_cache = False
    skip_db = True
    if not skip_db:
        try:
            signs = await check_cache(message)
            if (
                signs
                and signs != []
                and not any(
                    x["name"] in ["chat", "support", "positions"] for x in signs
                )
            ):
                using_cache = True
            message_id = await track_query(user_address, message, test=False)
        except Exception as e:
            print("cache error", e)
            message_id = -1
    else:
        message_id = -1
    if "switch" in message.lower().replace(",", "").split(
        " "
    ) or "change" in message.lower().replace(",", "").split(" "):
        return {
            "calls": [],
            "message": "To switch your connected chain, change the connected chain in your external wallet. However, if you specify chain names in future queries, those will be used! The connected chain is only used if a chain isn't specified in your query.",
            "message_id": message_id,
        }
    support_chat = False
    if not using_cache:
        available_functions = get_available_functions()
        idx = 0
        if "?" in message or message.startswith("what"):
            if "balance" in message.lower().replace("/", "").replace("s", "").replace(
                "?", ""
            ).replace(".", "").replace(",", "").split(" "):
                return {
                    "calls": [],
                    "message": "To check your balances, open the sidenav to your left.",
                    "message_id": message_id,
                }
            available_functions = {
                k: available_functions[k]
                for k in ("chat", "support")
                if k in available_functions
            }
            idx = -1
            support_chat = True
        (
            actions,
            protocols,
            tokens,
            chains,
            shortened_chains,
            keywords,
            pools,
            orderedtokens,
            confusing1,
            confusing2,
            confusing3,
        ) = await entities(fast=False, production=production)
        start_time = time.time()
        # print(1, time.time())
        rxtk = r"(?P<tkn>" + r"|".join(tokens - protocols - keywords) + r")"
        rxtk2 = r"(?P<tkn2>" + r"|".join(tokens - protocols - keywords) + r")"
        rxact = r"(?P<act>" + r"|".join(actions) + r")"
        rxch = r"(?P<chn>" + r"|".join(chains) + r")"
        (
            om,
            message,
            updated,
            error_messages,
        ) = await preprocessing(
            message,
            rxtk,
            rxtk2,
            rxact,
            rxch,
            keywords,
            tokens,
            protocols,
            chains,
            shortened_chains,
            pools,
            orderedtokens,
            actions,
            confusing1,
            confusing2,
            confusing3,
            support_chat,
        )
        # print(3, message)
        # sys.exit(0)
        if error_messages:
            error_message = ", ".join(error_messages)
            return {
                "calls": [],
                "message": error_message,
                "message_id": message_id,
            }
        if not support_chat:
            usable = []
            actions = list(available_functions.keys())
            ik = 0
            while ik < len(actions):
                action = actions[ik]
                if action in ["time", "condition", "chat", "support", "positions"]:
                    ik += 1
                    continue
                if action in message and action not in usable:
                    usable.append(action)
                ik += 1
            if "swap" not in usable:
                usable.append("swap")
            if "bridge" not in usable:
                usable.append("bridge")
            if "transfer" not in usable:
                usable.append("transfer")
            if "condition" not in usable:
                usable.append("condition")
            if "time" not in usable:
                usable.append("time")
            available_functions = {
                k: available_functions[k] for k in usable if k in available_functions
            }
            mid_time = time.time()
            # print(4, time.time())
            if mid_time - start_time > 1.5 * len(usable) - 6:
                print("error, slow processing", mid_time - start_time)
        results, response, idx = await run_conversation_retry(
            message,
            prompt,
            user_address,
            available_functions,
            idx,
            support_chat=support_chat,
        )
        # print(results)
        if not support_chat:
            if "condition" in message and "condition" not in [
                data.name for data in results
            ]:
                print("missed condition")
                message = message.replace("condition", "on condition")
                results, response, idx = await run_conversation_retry(
                    message, prompt, user_address, available_functions, idx
                )
                if "condition" in message and "condition" not in [
                    data.name for data in results
                ]:
                    print("missed condition again")
                    message = message.replace("on condition", "ON /CONDITION")
                    results, response, idx = await run_conversation_retry(
                        message, prompt, user_address, available_functions, idx
                    )
            if "buy" in message and "swap" not in [data.name for data in results]:
                print("missed swap")
                message = message.replace("buy", "/swap outputToken")
                results, response, idx = await run_conversation_retry(
                    message, prompt, user_address, available_functions, idx
                )
            if "sell" in message and "swap" not in [data.name for data in results]:
                print("missed swap")
                message = message.replace("sell", "/swap")
                results, response, idx = await run_conversation_retry(
                    message, prompt, user_address, available_functions, idx
                )
            ik = 0
            splitmsg = message.lower().replace("/", "").split(" ")
            while ik < len(actions):
                action = actions[ik]
                if action in [
                    "time",
                    "condition",
                    "chat",
                    "support",
                    "positions",
                    "buy",
                    "sell",
                ]:
                    ik += 1
                    continue
                if action in splitmsg and action not in [data.name for data in results]:
                    if action in ["long", "short"] and "close" in splitmsg:
                        ik += 1
                        continue
                    if action in ["borrow"] and "repay" in splitmsg:
                        ik += 1
                        continue
                    if action in ["transfer"] and "bridge" in [
                        data.name for data in results
                    ]:
                        ik += 1
                        continue
                    print(f"missed {action}")
                    message = " ".join(
                        [
                            x if x != action else f"/{action}"
                            for x in message.lower().split(" ")
                        ]
                    )
                    results, response, idx = await run_conversation_retry(
                        message, prompt, user_address, available_functions, idx
                    )
                    if idx == -1:
                        break
                    idx += 1
                else:
                    ik += 1
            # for p in protocols:
            # if p not in message.lower():
            # continue
            # found = False
            # for data in results:
            # for k, v in list(json.loads(data.arguments).items()):
            # if v == p:
            # found = True
            # break
            # if not found:
            # print(f"missed protocol {p}")
            # message = message.replace(p, f"protocolName {p}")
            # results, response = await run_conversation_retry(
            # message, prompt, user_address, available_functions
            # )
            # print('Human Response:', response)
            # print('Transactions to Sign:', results)
        signs, response = processing(
            results,
            response,
            message,
            updated,
            chains,
            protocols,
            tokens,
            shortened_chains,
            rxtk,
            rxtk2,
        )
        if len(signs) > 0:
            signs = postprocessing(
                signs, message, om, protocols, chains, tokens, default_chain
            )
        for s0 in signs:
            print(s0["name"], s0["args"])
        if (
            response
            and not isinstance(response, str)
            and not response.content
            and len(signs) > 0
        ):
            response.content = "Completed with calls to " + ", ".join(
                [s0["name"] for s0 in signs]
            )
        if (
            response
            and not isinstance(response, str)
            and response.content
            and len(signs) > 0
            and "with calls to" not in response.content
        ):
            names = [s0["name"] for s0 in signs]
            if (
                "chat" not in names
                and "support" not in names
                and "positions" not in names
            ):
                response.content = "Completed with calls to " + ", ".join(names)
        if save >= 0:
            save_entry(om, signs, save)
    if not skip_db:
        print(message_id)
        await track_response(message_id, signs, test=False)
    if not support_chat:
        verified = json.loads(await support())

        # perform processing only required for user visualization after saving entry and right before returning
        # since saved entries are used for AI training which has higher quality requirements
        protocol_to_chain = {p["name"]: p["chains"] for p in verified["protocols"]}
        pool_to_protocol = defaultdict(set)
        for p in verified["protocols"]:
            for pl in p["pools"].values():
                for pk in pl:
                    pool_to_protocol[pk].add(p["name"])
        all_protocols = set()
        all_chains = set()
        for p in verified["protocols"]:
            if "name" in p:
                if not p["name"].isdigit():
                    all_protocols.add(p["name"].lower())
        for c in verified["chains"]:
            all_chains.add(c.lower())

        signs = user_processing(
            signs, default_chain, protocol_to_chain, pool_to_protocol
        )

        end_time = time.time()
        # print(5, time.time())
        if not using_cache and end_time - mid_time > 1.5 * len(usable) - 6:
            print("error, slow ai", end_time - mid_time)

        # abort normal response if entity not supported
        for s0 in signs:
            name = s0["name"]
            args = s0["args"]
            if (
                "protocolName" in args
                and args["protocolName"] not in ["", "all"]
                and args["protocolName"] not in all_protocols
            ):
                return {
                    "calls": [],
                    "message": f"c4a1815b8349fc4c66463637d20d6a4d{args['protocolName'].capitalize()} is not supported on Slate yet!\n\nYou can find a list of supported protocols in the sidenav.\n\nPlease try again.",
                    "message_id": message_id,
                }
            if (
                "chainName" in args
                and args["chainName"] not in ["", "all"]
                and args["chainName"] not in all_chains
            ):
                return {
                    "calls": [],
                    "message": f"4d0ca75182638101abd452b9601b1966{args['chainName'].capitalize()} is not supported on Slate yet!\n\nYou can find a list of supported chains in the sidenav.\n\nPlease try again.",
                    "message_id": message_id,
                }
            if (
                "sourceChainName" in args
                and args["sourceChainName"] not in ["", "all"]
                and args["sourceChainName"] not in all_chains
            ):
                return {
                    "calls": [],
                    "message": f"4d0ca75182638101abd452b9601b1966{args['sourceChainName'].capitalize()} is not supported on Slate yet!\n\nYou can find a list of supported chains in the sidenav.\n\nPlease try again.",
                    "message_id": message_id,
                }
            if (
                "destinationChainName" in args
                and args["destinationChainName"] not in ["", "all"]
                and args["destinationChainName"] not in all_chains
            ):
                return {
                    "calls": [],
                    "message": f"4d0ca75182638101abd452b9601b1966{args['destinationChainName'].capitalize()} is not supported on Slate yet!\n\nYou can find a list of supported chains in the sidenav.\n\nPlease try again.",
                    "message_id": message_id,
                }

        # abort normal response if entity combo not supported
        chainmapping = chainnamestoids()
        for s0 in signs:
            name = s0["name"]
            args = s0["args"]
            # protocol-action combo not supported
            if (
                "protocolName" in args
                and args["protocolName"] not in ["", "all"]
                and args["protocolName"] not in verified["actions"][name]
            ):
                # inefficient methods of getting supported actions since all actions need to be cycled through and then checked for protocol support
                supported_actions = []
                for action, protocols in verified["actions"].items():
                    if args["protocolName"] in protocols:
                        supported_actions.append(f"* {action}")
                supported_actions_str = "\n".join(supported_actions)

                supported_protocols = []
                for protocol in verified["actions"][name]:
                    supported_protocols.append(f"* {protocol.capitalize()}")
                supported_protocols_str = "\n".join(supported_protocols)

                return {
                    "calls": [],
                    "message": f"ee885c90b5724715a08792234274f6b2{name.capitalize()} is not supported on {args['protocolName'].capitalize()}.\n\nThe following actions are supported on {args['protocolName'].capitalize()}:\n{supported_actions_str}\n\nYou can {name} on the following protocols:\n{supported_protocols_str}\n\nPlease try again.",
                    "message_id": message_id,
                }
            # protocol-chain combo not supported
            if (
                "protocolName" in args
                and "chainName" in args
                and args["protocolName"] not in ["", "all"]
                and args["chainName"] != ""
            ):
                for p in verified["protocols"]:
                    if (
                        p["name"] == args["protocolName"]
                        and args["chainName"].lower() not in p["chains"]
                    ):
                        if (
                            args["chainName"].lower() in ["polygon", "matic"]
                            and "matic" in p["chains"]
                        ):
                            continue
                        if (
                            args["chainName"].lower()
                            in [
                                "bnb smart chain",
                                "bsc",
                                "bnb",
                                "smart chain",
                                "binance",
                                "binancesmartchain",
                                "binance smart chain",
                                "bnbsmartchain",
                            ]
                            and "binance" in p["chains"]
                        ):
                            continue
                        # inefficient method of getting supported protocols since all protocols need to be cycled through and then checked for chain support
                        supported_protocols = []
                        for p2 in verified["protocols"]:
                            if args["chainName"].lower() in p2["chains"]:
                                supported_protocols.append(
                                    f"* {p2['name'].capitalize()}"
                                )
                        supported_protocols_str = "\n".join(supported_protocols)

                        supported_chains = []
                        for chain in p["chains"]:
                            supported_chains.append(f"* {chain.capitalize()}")
                        supported_chains_str = "\n".join(supported_chains)
                        return {
                            "calls": [],
                            "message": f"ef1f8f8ecb8841f7b6fceab5a2e1f22f{args['protocolName'].capitalize()} is not supported on {args['chainName'].capitalize()}.\n\nThe following protocols are supported on {args['chainName'].capitalize()}:\n{supported_protocols_str}\n\n{args['protocolName'].capitalize()} is currently supported on the following chains:\n{supported_chains_str}\n\nPlease try again.",
                            "message_id": message_id,
                        }

            # protocol-chain-pool combo not supported
            if (
                "protocolName" in args
                and "chainName" in args
                and "poolName" in args
                and args["poolName"] not in ["", "all"]
                and args["protocolName"] not in ["", "all"]
                and args["chainName"] != ""
                and args["chainName"].lower() in chainmapping
            ):
                for p in verified["protocols"]:
                    if (
                        p["name"] == args["protocolName"]
                        and str(chainmapping[args["chainName"].lower()]) in p["pools"]
                        and args["poolName"]
                        not in p["pools"][str(chainmapping[args["chainName"].lower()])]
                    ):
                        supported_pools = []
                        for pool in p["pools"][
                            str(chainmapping[args["chainName"].lower()])
                        ]:
                            supported_pools.append(f"* {pool}")
                        supported_pools_str = "\n".join(supported_pools)

                        return {
                            "calls": [],
                            "message": f"904da5fc885949d8890b0c578b70cfd5{args['poolName']} on {args['protocolName'].capitalize()} is not supported on {args['chainName'].capitalize()}.\n\nThese following {args['protocolName'].capitalize()} pools are supported on {args['chainName'].capitalize()}:\n{supported_pools_str}\n\nPlease try again.",
                            "message_id": message_id,
                        }
        for s0 in signs:
            name = s0["name"]
            args = s0["args"]
            if (
                name == "transfer"
                and "recipient" in args
                and args["recipient"].lower() == user_address.lower()
            ):
                return {
                    "calls": [],
                    "message": "Initially, you have to onboard funds onto your Slate account yourself. Transfer using your external wallet.",
                    "message_id": message_id,
                }
        gastoken = {
            "ethereum": "ETH",
            "optimism": "ETH",
            "bsc": "BNB",
            "gnosis": "xDAI",
            "polygon": "MATIC",
            "zksync": "ETH",
            "base": "ETH",
            "arbitrum": "ETH",
            "avalanche": "AVAX",
            "linea": "ETH",
            "blast": "ETH",
            "scroll": "ETH",
            "mode": "ETH",
            "manta": "ETH",
            "mantle": "MNT",
        }
        for ix in range(len(signs)):
            if ix == 0:
                continue
            if signs[ix - 1]["name"] in [
                "withdraw",
                "claim",
                "unstake",
                "unlock",
                "vote",
                "repay",
                "close",
            ]:
                continue
            chain = None
            if "chainName" in signs[ix]["args"]:
                chain = signs[ix]["args"]["chainName"]
            if "sourceChainName" in signs[ix]["args"]:
                chain = signs[ix]["args"]["sourceChainName"]
            if not chain:
                continue
            if (
                "chainName" in signs[ix - 1]["args"]
                and signs[ix - 1]["args"]["chainName"].lower() == chain.lower()
            ):
                if (
                    "amount" in signs[ix - 1]["args"]
                    and signs[ix - 1]["args"]["amount"] == "all"
                ) or (
                    "inputAmount" in signs[ix - 1]["args"]
                    and signs[ix - 1]["args"]["inputAmount"] == "all"
                ):
                    if (
                        "token" in signs[ix - 1]["args"]
                        and chain in gastoken
                        and signs[ix - 1]["args"]["token"].lower()
                        == gastoken[chain].lower()
                    ) or (
                        "inputToken" in signs[ix - 1]["args"]
                        and chain in gastoken
                        and isinstance(signs[ix - 1]["args"]["inputToken"], str)
                        and signs[ix - 1]["args"]["inputToken"].lower()
                        == gastoken[chain].lower()
                    ):
                        if (
                            signs[ix - 1]["name"] != "deposit"
                            and signs[ix]["name"] != "deposit"
                        ):
                            return {
                                "calls": [],
                                "message": f"You won't be able to perform this action sequence since you try to {signs[ix-1]['name']} all of your gas token on {chain} and don't have any left for the {signs[ix]['name']}!",
                                "message_id": message_id,
                            }
    if not using_cache:
        try:
            rmsg = response.content if response.content else response
        except Exception as e:
            print(3, e)
            rmsg = response if response else ""
        if any(x["name"] in ["chat", "support", "positions"] for x in signs):
            signs = [
                x for x in signs if x["name"] not in ["chat", "support", "positions"]
            ]
            if response:
                if isinstance(response, str):
                    try:
                        rmsg = response
                    except Exception as e:
                        print(response, signs, e)
                        rmsg = "Sorry, that's not supported yet!"
                        signs = []
                else:
                    try:
                        rmsg = response.content
                    except Exception as e:
                        print(response, signs, e)
                        rmsg = "Sorry, that's not supported yet!"
                        signs = []
            else:
                rmsg = "Sorry, that's not supported yet!"
                signs = []
        else:
            if response:
                if isinstance(response, str):
                    try:
                        rmsg = list(set(response.split("\n")))[0]
                    except Exception as e:
                        print(response, signs, e)
                        rmsg = "Sorry, that's not supported yet!"
                        signs = []
                else:
                    try:
                        rmsg = list(set(response.content.split("\n")))[0]
                    except Exception as e:
                        print(response, signs, e)
                        rmsg = "Sorry, that's not supported yet!"
                        signs = []
            else:
                rmsg = "Sorry, that's not supported yet!"
                signs = []
    else:
        rmsg = "Completed with calls to " + ", ".join([c["name"] for c in signs])
    if not support_chat:
        rmsg = rmsg.replace("functions.", "")
    return {
        "calls": signs,
        "message": rmsg,
        "message_id": message_id,
    }


def save_entry(om, signs, save):
    num = save
    with open(f"test/res{num}.json", "r") as f:
        sofar = json.load(f)
    lk = {x.lower(): x for x in list(sofar.keys())}
    if om in lk:
        sofar[lk[om]] = signs
    else:
        sofar[om] = signs
    with open(f"test/res{num}.json", "w") as f:
        json.dump(sofar, f)


def user_processing(signs, default_chain, protocol_to_chain, pool_to_protocol):
    signs = assign_all(signs)
    signs = process_add_value_units(signs)
    signs = process_all_secondary(signs)
    signs = add_defaults(signs, default_chain, protocol_to_chain, pool_to_protocol)
    signs = chain_aliases(signs)
    # signs = pendle_pool(signs)
    return signs


def pendle_pool(signs):
    """
    Ensure Pendle pool names are standardized
    """
    processed = []
    for ix in range(len(signs)):
        if (
            "poolName" in signs[ix]["args"]
            and "protocolName" in signs[ix]["args"]
            and signs[ix]["args"]["protocolName"] == "pendle"
        ):
            if signs[ix]["args"]["poolName"] != "" and signs[ix]["args"]["poolName"][
                :3
            ] not in ["sy-", "yt-", "pt-"]:
                signs[ix]["args"]["poolName"] = "sy-" + signs[ix]["args"]["poolName"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def chain_aliases(signs):
    """
    Ensure chains with many aliases are standardized
    """
    processed = []
    for ix in range(len(signs)):
        chainkeys = ["chainName", "sourceChainName", "destinationChainName"]
        for ck in chainkeys:
            if ck in signs[ix]["args"] and signs[ix]["args"][ck].lower() in [
                "bnb smart chain",
                "bsc",
                "bnb",
                "smart chain",
                "binance",
                "binancesmartchain",
                "binance smart chain",
                "bnbsmartchain",
            ]:
                signs[ix]["args"][ck] = "bsc"
            if ck in signs[ix]["args"] and signs[ix]["args"][ck].lower() in [
                "polygon",
                "matic",
            ]:
                signs[ix]["args"][ck] = "polygon"
            if ck in signs[ix]["args"] and signs[ix]["args"][ck].lower() in [
                "mainnet",
                "homestead",
                "ethereum",
            ]:
                signs[ix]["args"][ck] = "ethereum"
            if ck in signs[ix]["args"] and signs[ix]["args"][ck].lower() in [
                "zksync",
                "zksync era",
            ]:
                signs[ix]["args"][ck] = "zksync"
            if ck in signs[ix]["args"] and signs[ix]["args"][ck].lower() in [
                "arbitrum",
                "arbitrum one",
            ]:
                signs[ix]["args"][ck] = "arbitrum"
            if ck in signs[ix]["args"] and signs[ix]["args"][ck].lower() in [
                "op mainnet",
                "optimism",
            ]:
                signs[ix]["args"][ck] = "optimism"
            if ck in signs[ix]["args"] and signs[ix]["args"][ck].lower() in [
                "linea",
                "linea-mainnet",
            ]:
                signs[ix]["args"][ck] = "linea"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def add_defaults(signs, default_chain, protocol_to_chain, pool_to_protocol):
    """
    Add default values to empty parameters
    """
    with open("tools-min.json", "r") as f:
        data = json.load(f)
    allowed_keys = {}
    for d in data:
        allowed_keys[d["function"]["name"]] = list(
            d["function"]["parameters"]["properties"]
        )
    allowed_keys["swap"].append("outputAmount")
    allowed_keys["swap"].append("side")
    allowed_keys["condition"].append("type")
    allowed_keys["condition"].append("value_token")
    allowed_keys["condition"].append("value_units")
    allowed_keys["stake"].append("poolName")
    processed = []
    for ix in range(len(signs)):
        for key in allowed_keys.get(signs[ix]["name"], []):
            if key not in signs[ix]["args"] or signs[ix]["args"][key] == "":
                if key == "amount":
                    signs[ix]["args"][key] = ""
                if key == "outputToken":
                    if signs[ix]["name"] in ["swap"]:
                        signs[ix]["args"][key] = "usdc"
                    else:
                        signs[ix]["args"][key] = ""
                if key == "slippage":
                    signs[ix]["args"][key] = ""
                if key == "sourceChainName":
                    if (
                        "protocolName" in signs[ix]["args"]
                        and signs[ix]["args"]["protocolName"] not in ["", "all"]
                        and len(protocol_to_chain[signs[ix]["args"]["protocolName"]])
                        == 1
                    ):
                        signs[ix]["args"][key] = protocol_to_chain[
                            signs[ix]["args"]["protocolName"]
                        ][0]
                    else:
                        signs[ix]["args"]["sourceChainName"] = default_chain
                if key == "destinationChainName":
                    signs[ix]["args"][key] = ""
                if key == "chainName":
                    if (
                        "protocolName" in signs[ix]["args"]
                        and signs[ix]["args"]["protocolName"] not in ["", "all"]
                        and signs[ix]["args"]["protocolName"] in protocol_to_chain
                        and len(protocol_to_chain[signs[ix]["args"]["protocolName"]])
                        == 1
                    ):
                        signs[ix]["args"][key] = protocol_to_chain[
                            signs[ix]["args"]["protocolName"]
                        ][0]
                    else:
                        if signs[ix]["name"] in ["long", "short", "close"]:
                            signs[ix]["args"]["protocolName"] = "gmx"
                            signs[ix]["args"][key] = "arbitrum"
                        else:
                            signs[ix]["args"][key] = default_chain
                if key == "protocolName":
                    if (
                        "poolName" in signs[ix]["args"]
                        and signs[ix]["args"]["poolName"] not in ["", "all"]
                        and len(pool_to_protocol[signs[ix]["args"]["poolName"]]) == 1
                    ):
                        signs[ix]["args"][key] = list(
                            pool_to_protocol[signs[ix]["args"]["poolName"]]
                        )[0]
                    elif signs[ix]["name"] not in ["swap", "bridge", "transfer"]:
                        if signs[ix]["name"] in ["long", "short", "close"]:
                            signs[ix]["args"][key] = "gmx"
                            if (
                                "chainName" not in signs[ix]["args"]
                                or signs[ix]["args"]["chainName"] == ""
                            ):
                                signs[ix]["args"]["chainName"] = "arbitrum"
                        else:
                            signs[ix]["args"][key] = ""
                if key == "poolName":
                    if "protocolName" in signs[ix]["args"] and signs[ix]["args"][
                        "protocolName"
                    ] in ["pendle"]:
                        signs[ix]["args"][key] = ""
            if key in signs[ix]["args"] and key == "protocolName":
                if signs[ix]["name"] in ["long", "short", "close"]:
                    if signs[ix]["args"][key] not in ["gmx", "hyperliquid"]:
                        signs[ix]["args"]["protocolName"] = "gmx"

        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def assign_all(signs):
    """
    Assign "all" to empty token/amount values to avoid ambiguity
    """

    # to inputToken and inputAmount for swaps
    processed = []
    for ix in range(len(signs)):
        s0 = signs[ix]
        if (
            ix == 0
            and s0["name"] in ["swap"]
            and "inputToken" not in s0["args"]
            and "inputAmountUnits" not in s0["args"]
        ):
            if "outputToken" in s0["args"]:
                s0["args"]["inputToken"] = "eth"
            else:
                s0["args"]["inputToken"] = "all"
        if (
            "swap" in s0["name"]
            and "inputToken" in s0["args"]
            and (
                "inputAmount" not in s0["args"]
                or (
                    "inputAmount" in s0["args"]
                    and isinstance(s0["args"]["inputAmount"], str)
                    and s0["args"]["inputAmount"].lower() == "outputAmount".lower()
                )
            )
            and "outputAmount" not in s0["args"]
        ):
            if ix == 0:
                s0["args"]["inputAmount"] = "all"
            else:
                update = True
                for iy in range(ix):
                    if signs[iy]["name"] not in [
                        "condition",
                        "time",
                        "chat",
                        "support",
                        "positions",
                    ]:
                        update = False
                        break
                if update:
                    s0["args"]["inputAmount"] = "all"
        processed.append(s0)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    # to token and amount for other actions
    processed = []
    for ix in range(len(signs)):
        s0 = signs[ix]
        if (
            ix == 0
            and s0["name"]
            not in [
                "swap",
                "long",
                "close",
                "short",
                "claim",
                "vote",
                "chat",
                "support",
                "positions",
                "condition",
                "time",
            ]
            and "token" not in s0["args"]
            and "amount_units" not in s0["args"]
        ):
            s0["args"]["token"] = "all"
        if "token" in s0["args"] and (
            "amount" not in s0["args"]
            or (
                "amount" in s0["args"]
                and isinstance(s0["args"]["amount"], str)
                and s0["args"]["amount"].lower() == "outputAmount".lower()
            )
        ):
            if ix == 0:
                s0["args"]["amount"] = "all"
            else:
                update = True
                for iy in range(ix):
                    if signs[iy]["name"] not in [
                        "condition",
                        "time",
                        "chat",
                        "support",
                        "positions",
                    ]:
                        update = False
                        break
                if update:
                    s0["args"]["amount"] = "all"
        processed.append(s0)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    return signs


def postprocessing(
    signs, message: str, om: str, protocols, chains, tokens, default_chain
):
    """
    Comb through and edit completed calls before final response with different independent checks
    """
    signs = process_empty_protocol_names_forwards(signs)
    signs = process_empty_protocol_names_backwards(signs)
    signs = process_nested_conditions(signs)
    signs = process_token_amount_assignment(signs, message)
    signs = process_zero_condition(signs)
    signs = process_bad_condition(signs, message)
    signs = process_price_condition(signs)
    signs = process_price_condition_2(signs)
    signs = process_double_bridge(signs)
    signs = process_protocol_transfer_mistake(signs, protocols)
    signs = process_chain_transfer_mistake(signs, chains)
    signs = process_all_after_amount(signs)
    signs = process_duplicate_chain_name(signs)
    signs = process_bridge_chain_name(signs, default_chain)
    signs = process_percentages_and_multiples(signs, message)
    signs = process_word_numbers(signs)
    signs = process_time(signs, message)
    signs = process_output(signs)
    signs = process_protocol_token(signs, tokens)
    signs = process_repeats(signs)
    signs = process_output_token(signs)
    signs = process_extra_all(signs, message, om)
    signs = process_lsc_all(signs)
    signs = process_missing_leverage(signs, message)
    signs = process_first_output_token(signs)
    # signs = process_mistaken_amount(signs, om, message)
    signs = process_mistaken_rewards(signs, message)
    signs = process_all_output_amount(signs)
    signs = process_all_chain(signs, message)
    signs = process_double_condition(signs)
    signs = process_missing_amount_units(signs, om)
    signs = process_cross_chain_swap(signs, message, tokens, protocols)
    signs = process_same_token_swap_perp(signs)
    signs = process_same_chain_bridge(signs)
    signs = process_incorrect_params(signs)
    signs = process_output_amount_all(signs)
    signs = process_output_token_all(signs)
    signs = process_gmx_ambiguity(signs)
    return signs


def process_gmx_ambiguity(signs):
    """
    Remove gmx protocol if its the token
    """
    processed = []
    for ix in range(len(signs)):
        if (
            "outputToken" in signs[ix]["args"]
            and signs[ix]["args"]["outputToken"] == "gmx"
            and "protocolName" in signs[ix]["args"]
            and signs[ix]["args"]["protocolName"] == "gmx"
        ):
            del signs[ix]["args"]["protocolName"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_output_token_all(signs):
    """
    Remove output token if it is "all"
    """
    processed = []
    for ix in range(len(signs)):
        if (
            "outputToken" in signs[ix]["args"]
            and signs[ix]["args"]["outputToken"] == "all"
        ):
            del signs[ix]["args"]["outputToken"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_word_numbers(signs):
    """
    Convert words of numbers to actual numbers
    """
    processed = []
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] == "condition"
            and "value" in signs[ix]["args"]
            and "value_token" in signs[ix]["args"]
            and signs[ix]["args"]["value_token"]
            in ["thousand", "million", "billion", "trillion"]
        ):
            if signs[ix]["args"]["value_token"] == "thousand":
                signs[ix]["args"]["value"] = str(
                    float(signs[ix]["args"]["value"]) * 1000
                )
            if signs[ix]["args"]["value_token"] == "million":
                signs[ix]["args"]["value"] = str(
                    float(signs[ix]["args"]["value"]) * 1000000
                )
            if signs[ix]["args"]["value_token"] == "billion":
                signs[ix]["args"]["value"] = str(
                    float(signs[ix]["args"]["value"]) * 1000000000
                )
            if signs[ix]["args"]["value_token"] == "trillion":
                signs[ix]["args"]["value"] = str(
                    float(signs[ix]["args"]["value"]) * 1000000000000
                )
            del signs[ix]["args"]["value_token"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_all_chain(signs, message):
    """
    Assign all to chain name if necessary
    """
    processed = []
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] == "bridge"
            and (
                "sourceChainName" not in signs[ix]["args"]
                or signs[ix]["args"]["sourceChainName"] == ""
            )
            and "all chain" in message.lower()
        ):
            signs[ix]["args"]["sourceChainName"] = "all"
        if (
            signs[ix]["name"] not in ["bridge", "support", "condition", "time"]
            and (
                "chainName" not in signs[ix]["args"]
                or signs[ix]["args"]["chainName"] == ""
            )
            and "all chain" in message.lower()
        ):
            signs[ix]["args"]["chainName"] = "all"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_all_secondary(signs):
    """
    Assign all to secondary actions if necessary
    """
    processed = [signs[0]] if len(signs) > 0 else []
    for ix in range(1, len(signs)):
        if (
            signs[ix]["name"] == "swap"
            and signs[0]["name"] == "swap"
            and "outputToken" in signs[0]["args"]
            and "outputToken" in signs[ix]["args"]
            and signs[ix]["args"]["outputToken"] == signs[0]["args"]["outputToken"]
            and "inputAmount" not in signs[ix]["args"]
            and "inputAmount" in signs[0]["args"]
            and signs[0]["args"]["inputAmount"] == "all"
        ):
            signs[ix]["args"]["inputAmount"] = "all"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_protocol_token(signs, tokens):
    """
    Assign token to protocol name value if necessary
    """
    processed = []
    for ix in range(len(signs)):
        if signs[ix]["name"] == "swap":
            if (
                "protocolName" in signs[ix]["args"]
                and "inputToken" not in signs[ix]["args"]
                and signs[ix]["args"]["protocolName"] in tokens
            ):
                signs[ix]["args"]["inputToken"] = signs[ix]["args"]["protocolName"]
                del signs[ix]["args"]["protocolName"]
            if (
                "protocolName" in signs[ix]["args"]
                and "outputToken" not in signs[ix]["args"]
                and signs[ix]["args"]["protocolName"] in tokens
            ):
                signs[ix]["args"]["outputToken"] = signs[ix]["args"]["protocolName"]
                del signs[ix]["args"]["protocolName"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_output_amount_all(signs):
    """
    Update all to outputAmount if necessary
    """
    processed = []
    for ix in range(len(signs)):
        if ix == 0:
            processed.append(signs[ix])
            continue
        if "outputToken" in signs[ix - 1]["args"] and signs[ix - 1]["name"] not in [
            "long",
            "short",
            "close",
        ]:
            if "token" in signs[ix]["args"] and "amount" in signs[ix]["args"]:
                if (
                    signs[ix]["args"]["token"] == signs[ix - 1]["args"]["outputToken"]
                    and signs[ix]["args"]["amount"] == "all"
                ):
                    signs[ix]["args"]["amount"] = "outputAmount"
            if "inputToken" in signs[ix]["args"] and "inputAmount" in signs[ix]["args"]:
                if (
                    signs[ix]["args"]["inputToken"]
                    == signs[ix - 1]["args"]["outputToken"]
                    and signs[ix]["args"]["inputAmount"] == "all"
                ):
                    signs[ix]["args"]["inputAmount"] = "outputAmount"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_cross_chain_swap(signs, message, tokens, protocols):
    """
    Add a swap after a bridge if a user wants to do a cross chain swap
    """
    processed = []
    added = 0
    toadd = []
    msglist = message.replace("'", "").split(" ")
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] == "bridge"
            and "destinationChainName" in signs[ix]["args"]
        ):
            try:
                idx = msglist.index(signs[ix]["args"]["destinationChainName"])
                if msglist[idx + 1] == "chain" and (
                    (msglist[idx + 2] in tokens and msglist[idx + 2] not in protocols)
                    or msglist[idx + 3] == "token"
                ):
                    potential = [
                        ix + 1,
                        {
                            "name": "swap",
                            "args": {
                                "inputAmount": "outputAmount",
                                "inputToken": signs[ix]["args"]["token"],
                                "outputToken": msglist[idx + 2],
                                "chainName": signs[ix]["args"]["destinationChainName"],
                            },
                        },
                    ]
                    if ix + 1 >= len(signs) or (
                        ix + 1 < len(signs) and signs[ix + 1] != potential[1]
                    ):
                        toadd.append(potential)
            except Exception:
                pass
        processed.append(signs[ix])
    toadd = toadd[::-1]
    for t in toadd:
        if t[0] < len(processed):
            processed.insert(t[0], t[1])
        else:
            processed.append(t[1])
        added += 1
    assert len(signs) + added == len(processed), (signs, processed, added)
    signs = processed
    return signs


def process_missing_leverage(signs, message):
    """
    Add leverage if it missing
    """
    processed = []
    msglist = message.replace("'", "").split(" ")
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] == "close"
            and "leverageMultiplier" not in signs[ix]["args"]
            and "inputToken" in signs[ix]["args"]
        ):
            try:
                idx = msglist.index(signs[ix]["args"]["inputToken"])
                if msglist[idx - 1] == "leverage":
                    signs[ix]["args"]["leverageMultiplier"] = msglist[idx - 2]
            except Exception as e:
                print(5, e)
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_double_bridge(signs):
    """
    Consolidate a double bridge into a single if using a protocol
    """
    processed = []
    removed = 0
    for ix in range(len(signs) - 1):
        if "skip" in signs[ix]:
            removed += 1
            continue
        if (
            signs[ix]["name"] in ["bridge"]
            and signs[ix + 1]["name"] in ["bridge"]
            and "token" in signs[ix]["args"]
            and "token" in signs[ix + 1]["args"]
            and "destinationChainName" in signs[ix]["args"]
            and "sourceChainName" in signs[ix + 1]["args"]
            and "destinationChainName" in signs[ix + 1]["args"]
            and "protocolName" in signs[ix + 1]["args"]
            and "protocolName" not in signs[ix]["args"]
            and signs[ix]["args"]["token"] == signs[ix + 1]["args"]["token"]
            and signs[ix]["args"]["destinationChainName"]
            == signs[ix + 1]["args"]["sourceChainName"]
            and signs[ix + 1]["args"]["sourceChainName"]
            == signs[ix + 1]["args"]["destinationChainName"]
        ):
            signs[ix]["args"]["protocolName"] = signs[ix + 1]["args"]["protocolName"]
            signs[ix + 1]["skip"] = True
        processed.append(signs[ix])
    if "skip" not in signs[-1]:
        processed.append(signs[-1])
    else:
        removed += 1
    assert len(signs) == len(processed) + removed, (signs, processed, removed)
    signs = processed
    return signs


def process_lsc_all(signs):
    """
    Remove protocolName from actions if its "all"
    """
    processed = []
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] in ["long", "short", "close", "swap", "bridge"]
            and "protocolName" in signs[ix]["args"]
            and signs[ix]["args"]["protocolName"] == "all"
        ):
            del signs[ix]["args"]["protocolName"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_double_condition(signs):
    """
    Separate combined conditions
    """
    processed = []
    added = 0
    for ix in range(len(signs)):
        new = {}
        if (
            signs[ix]["name"] in ["condition"]
            and "subject_2" in signs[ix]["args"]
            and "comparator_2" in signs[ix]["args"]
            and "value_2" in signs[ix]["args"]
            and "type_2" in signs[ix]["args"]
        ):
            new = {
                "name": "condition",
                "args": {
                    "subject": signs[ix]["args"]["subject_2"],
                    "comparator": signs[ix]["args"]["comparator_2"],
                    "value": signs[ix]["args"]["value_2"],
                    "type": signs[ix]["args"]["type_2"],
                },
            }
            added += 1
            del signs[ix]["args"]["subject_2"]
            del signs[ix]["args"]["comparator_2"]
            del signs[ix]["args"]["value_2"]
            del signs[ix]["args"]["type_2"]
        processed.append(signs[ix])
        if new != {}:
            processed.append(new)
    assert len(signs) + added == len(processed), (signs, processed, added)
    signs = processed
    return signs


def process_missing_amount_units(signs, om):
    """
    Add amount units if clearly there but not populated
    """
    processed = []
    msglist = [x for x in om.split(" ") if x != "of"]
    for ix in range(len(signs)):
        if (
            "amount_units" not in signs[ix]["args"]
            and "amount" in signs[ix]["args"]
            and isinstance(signs[ix]["args"]["amount"], str)
            and "token" in signs[ix]["args"]
            and signs[ix]["args"]["token"] != "usd"
        ):
            if (
                msglist.count(signs[ix]["args"]["amount"]) == 1
                and "$" + signs[ix]["args"]["amount"] in msglist
            ):
                signs[ix]["args"]["amount_units"] = "usd"
        if (
            "inputAmountUnits" not in signs[ix]["args"]
            and "inputAmount" in signs[ix]["args"]
            and isinstance(signs[ix]["args"]["inputAmount"], str)
            and "inputToken" in signs[ix]["args"]
            and isinstance(signs[ix]["args"]["inputToken"], str)
            and signs[ix]["args"]["inputToken"] != "usd"
        ):
            if (
                msglist.count(signs[ix]["args"]["inputAmount"]) == 1
                and "$" + signs[ix]["args"]["inputAmount"] in msglist
            ):
                signs[ix]["args"]["inputAmountUnits"] = "usd"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_first_output_token(signs):
    """
    Remove outputToken if first call
    """
    processed = []
    for ix in range(len(signs)):
        if ix == 0:
            k = next(
                (
                    key
                    for key, value in signs[ix]["args"].items()
                    if value == "outputToken"
                ),
                None,
            )
            if k:
                del signs[ix]["args"][k]
        else:
            truefirst = True
            for iy in range(ix):
                if signs[iy]["name"] not in [
                    "condition",
                    "time",
                    "chat",
                    "support",
                    "positions",
                ]:
                    truefirst = False
            if truefirst:
                k = next(
                    (
                        key
                        for key, value in signs[ix]["args"].items()
                        if value == "outputToken"
                    ),
                    None,
                )
                if k:
                    del signs[ix]["args"][k]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_incorrect_params(signs):
    """
    Remove incorrect params
    """
    with open("tools-min.json", "r") as f:
        data = json.load(f)
    allowed_keys = {}
    for d in data:
        allowed_keys[d["function"]["name"]] = list(
            d["function"]["parameters"]["properties"]
        )
    allowed_keys["swap"].append("outputAmount")
    allowed_keys["swap"].append("side")
    allowed_keys["condition"].append("type")
    allowed_keys["condition"].append("value_token")
    allowed_keys["condition"].append("value_units")
    allowed_keys["stake"].append("poolName")
    processed = []
    for ix in range(len(signs)):
        if signs[ix]["name"] in allowed_keys:
            try:
                signs[ix]["args"] = {
                    k: v
                    for k, v in signs[ix]["args"].items()
                    if k in allowed_keys[signs[ix]["name"]]
                }
            except Exception as e:
                print(signs)
                raise e
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_all_output_amount(signs):
    """
    Remove outputAmount if all or outputAmount
    """
    processed = []
    for ix in range(len(signs)):
        if "outputAmount" in signs[ix]["args"] and signs[ix]["args"][
            "outputAmount"
        ] in ["all", "outputAmount"]:
            del signs[ix]["args"]["outputAmount"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_mistaken_rewards(signs, message):
    """
    Set token to outputToken if incorrectly assigned to rewards token
    """
    processed = []
    msglist0 = [
        x.replace("$", "").replace(",", "") for x in message.split(" ") if x != "of"
    ]
    for ix in range(len(signs)):
        if ix == 0:
            processed.append(signs[ix])
            continue
        keys = ["token", "inputToken"]
        for key in keys:
            if key in signs[ix]["args"] and signs[ix - 1]["name"] == "claim":
                target = signs[ix]["args"][key]
                idcs = [i + 1 for i, x in enumerate(msglist0) if x == target]
                nexts = [msglist0[idc].lower() for idc in idcs]
                if "rewards" in nexts:
                    signs[ix]["args"][key] = "outputToken"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    return signs


def process_mistaken_amount(signs, om, message):
    """
    Set amount to outputAmount if incorrectly assigned
    """
    processed = []
    omsglist0 = [x.replace("$", "") for x in om.split(" ") if x not in ["of", "my"]]
    msglist0 = [x.replace("$", "") for x in message.split(" ") if x not in ["of", "my"]]
    for ix in range(len(signs)):
        if ix == 0:
            processed.append(signs[ix])
            continue
        keys = [["token", "amount"], ["inputToken", "inputAmount"]]
        for key in keys:
            if key[0] in signs[ix]["args"] and key[1] in signs[ix]["args"]:
                if not isinstance(signs[ix]["args"][key[0]], str):
                    continue
                if not isinstance(signs[ix]["args"][key[1]], str):
                    continue
                if signs[ix]["args"][key[1]] in ["all", "half", "outputAmount"]:
                    continue
                if "%" in signs[ix]["args"][key[1]]:
                    continue
                if (
                    "amount_units" in signs[ix]["args"]
                    or "inputAmountUnits" in signs[ix]["args"]
                ):
                    continue
                if (
                    "poolName" in signs[ix]["args"]
                    and signs[ix]["args"][key[0]] in signs[ix]["args"]["poolName"]
                ):
                    continue
                if signs[ix]["args"][key[0]] == "usd":
                    continue
                target = signs[ix]["args"][key[1]]
                midcs = [
                    i + 1
                    for i, x in enumerate(msglist0)
                    if x == target or x == f"'{target}'"
                ]
                mnexts = [
                    msglist0[midc].lower() for midc in midcs if midc != len(msglist0)
                ]
                oidcs = [
                    i + 1
                    for i, x in enumerate(omsglist0)
                    if x == target or x == f"'{target}'"
                ]
                onexts = [
                    omsglist0[oidc].lower() for oidc in oidcs if oidc != len(omsglist0)
                ]
                nexts = mnexts + onexts
                # print('n', nexts, target)
                sl = signs[ix]["args"][key[0]].lower()
                if sl not in nexts and f"{sl}." not in nexts and f"{sl}," not in nexts:
                    signs[ix]["args"][key[1]] = "outputAmount"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_extra_all(signs, message, om):
    """
    Update "all" value for amounts if hallucinated
    """
    processed = []
    for ix in range(len(signs)):
        if ix == 0:
            processed.append(signs[ix])
            continue
        truefirst = True
        for iy in range(ix):
            if signs[iy]["name"] not in [
                "condition",
                "time",
                "chat",
                "support",
                "positions",
            ]:
                truefirst = False
        if truefirst:
            processed.append(signs[ix])
            continue
        keys = ["amount", "inputAmount"]
        for key in keys:
            if (
                key in signs[ix]["args"]
                and isinstance(signs[ix]["args"][key], str)
                and signs[ix]["args"][key] == "all"
                and "all" not in message.lower().split(" ")
                and "everything" not in message.lower().split(" ")
                and "all" not in om.lower().split(" ")
                and "everything" not in om.lower().split(" ")
            ):
                signs[ix]["args"][key] = "outputAmount"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_output_token(signs):
    """
    Update outputToken
    """
    processed = []
    for ix in range(len(signs)):
        if ix == 0:
            processed.append(signs[ix])
            continue
        keys = ["token", "inputToken"]
        keys2 = ["token", "outputToken"]
        for key in keys:
            if (
                key in signs[ix]["args"]
                and signs[ix]["args"][key] == "outputToken"
                and "poolName" not in list(signs[ix]["args"].keys())
            ):
                for key2 in keys2:
                    if (
                        key2 in signs[ix - 1]["args"]
                        and signs[ix - 1]["name"]
                        in ["swap", "bridge", "withdraw", "borrow"]
                        and signs[ix - 1]["args"][key2] not in ["liquidity"]
                    ):
                        signs[ix]["args"][key] = signs[ix - 1]["args"][key2]
                        break
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_bad_condition(signs, message):
    """
    remove incorrect conditions
    """
    processed = []
    skipped = 0
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] in ["condition"]
            and "value" in signs[ix]["args"]
            and signs[ix]["args"]["value"]
            not in message.replace("protocol", "").replace("  ", " ")
            and "x" not in signs[ix]["args"]["value"]
            and "%" not in signs[ix]["args"]["value"]
        ):
            try:
                _y = float(signs[ix]["args"]["value"].replace(",", "").replace(".", ""))
            except Exception:
                # print(e)
                skipped += 1
                continue
        if (
            signs[ix]["name"] in ["condition"]
            and "type" in signs[ix]["args"]
            and signs[ix]["args"]["type"] in ["balance"]
            and "balance" not in message
            and "rewards" not in message
            and "holding" not in message
            and "tvl" not in message
            and "aum" not in message
        ):
            skipped += 1
            continue
        if signs[ix]["name"] in ["condition", "time"] and signs[ix]["args"] == {}:
            skipped += 1
            continue
        processed.append(signs[ix])
    assert len(signs) == len(processed) + skipped, (signs, processed, skipped)
    signs = processed
    return signs


def process_percentages_and_multiples(signs, message):
    """
    Add percentage sign or multiplication sign to certain condition types and slippage
    """
    processed = []
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] in ["condition"]
            and "value" in signs[ix]["args"]
            and "type" in signs[ix]["args"]
            and signs[ix]["args"]["type"] in ["ltv"]
        ):
            if "%" not in signs[ix]["args"]["value"]:
                signs[ix]["args"]["value"] = signs[ix]["args"]["value"] + "%"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    processed = []
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] in ["condition"]
            and "value" in signs[ix]["args"]
            and "type" in signs[ix]["args"]
            and signs[ix]["args"]["type"]
            in ["price", "market cap", "gas", "fdv", "balance"]
        ):
            if signs[ix]["args"]["value"] not in message.replace(",", "").replace(
                ".", ""
            ).replace("!", "").replace(";", "").split(" ") and signs[ix]["args"][
                "value"
            ] + "x" in message.replace(",", "").replace(".", "").replace(
                "!", ""
            ).replace(";", "").split(" "):
                signs[ix]["args"]["value"] = signs[ix]["args"]["value"] + "x"
                if (
                    "value_token" in signs[ix]["args"]
                    and signs[ix]["args"]["value_token"] == "x"
                ):
                    del signs[ix]["args"]["value_token"]
                if (
                    "value_units" in signs[ix]["args"]
                    and signs[ix]["args"]["value_units"] == "x"
                ):
                    del signs[ix]["args"]["value_units"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    processed = []
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] in ["condition"]
            and "value" in signs[ix]["args"]
            and "type" in signs[ix]["args"]
            and signs[ix]["args"]["type"]
            in ["price", "market cap", "gas", "fdv", "balance"]
        ):
            if signs[ix]["args"]["value"] not in message.replace(",", "").replace(
                ".", ""
            ).replace("!", "").replace(";", "").split(" ") and signs[ix]["args"][
                "value"
            ] + "%" in message.replace(",", "").replace(".", "").replace(
                "!", ""
            ).replace(";", "").split(" "):
                signs[ix]["args"]["value"] = signs[ix]["args"]["value"] + "%"
                if (
                    "value_token" in signs[ix]["args"]
                    and signs[ix]["args"]["value_token"] == "%"
                ):
                    del signs[ix]["args"]["value_token"]
                if (
                    "value_units" in signs[ix]["args"]
                    and signs[ix]["args"]["value_units"] == "%"
                ):
                    del signs[ix]["args"]["value_units"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    processed = []
    for ix in range(len(signs)):
        if "slippage" in signs[ix]["args"]:
            if "%" not in signs[ix]["args"]["slippage"]:
                signs[ix]["args"]["slippage"] = signs[ix]["args"]["slippage"] + "%"
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_bridge_chain_name(signs, default_chain):
    """
    Add chain name to subsequent calls after a bridge if not there
    """
    processed = []
    for ix in range(len(signs)):
        if (
            signs[ix]["name"] in ["bridge"]
            and "destinationChainName" in signs[ix]["args"]
        ):
            for iy in range(ix + 1, len(signs)):
                if signs[iy]["name"] in ["bridge"]:
                    if (
                        "sourceChainName" not in signs[iy]["args"]
                        and signs[ix]["args"]["destinationChainName"]
                        != signs[iy]["args"]["destinationChainName"]
                    ):
                        if not default_chain:
                            signs[iy]["args"]["sourceChainName"] = signs[ix]["args"][
                                "destinationChainName"
                            ]
                    break
                if (
                    signs[iy]["name"]
                    not in ["condition", "time", "chat", "support", "positions"]
                    and "chainName" not in signs[iy]["args"]
                ):
                    signs[iy]["args"]["chainName"] = signs[ix]["args"][
                        "destinationChainName"
                    ]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    processed = []
    for ix in range(len(signs)):
        if signs[ix]["name"] in ["bridge"] and (
            "sourceChainName" not in signs[ix]["args"]
            or signs[ix]["args"]["sourceChainName"] == ""
        ):
            for iy in range(ix - 1, -1, -1):
                if "chainName" in signs[iy]["args"]:
                    signs[ix]["args"]["sourceChainName"] = signs[iy]["args"][
                        "chainName"
                    ]
                    break
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_add_value_units(signs):
    """
    Add value_units to unlabeled condition values for UI visualization
    """
    processed = []
    for s in signs:
        if s["name"] in ["condition"]:
            if (
                "value" in s["args"]
                and "subject" in s["args"]
                and "value_units" not in s["args"]
                and "value_token" not in s["args"]
                and "type" in s["args"]
                and s["args"]["type"] not in ["gas", "balance", "yield"]
                and "%" not in s["args"]["value"]
                and "/" not in s["args"]["subject"]
                and "-" not in s["args"]["subject"]
            ):
                s["args"]["value_units"] = "usd"
        processed.append(s)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_zero_condition(signs):
    """
    Remove 0 value conditions
    """
    processed = []
    skipped = 0
    for i in range(len(signs)):
        s = signs[i]
        if "value" in s["args"] and s["args"]["value"] == "0":
            skipped += 1
            continue
        processed.append(s)
    assert len(signs) == len(processed) + skipped, (signs, processed, skipped)
    signs = processed
    return signs


def process_output(signs) -> list[dict[str, Any]]:
    """
    Standardize missing output
    """
    processed = []
    for i in range(len(signs)):
        s = signs[i]
        if i > 0:
            p = signs[i - 1]
            truefirst = True
            for j in range(i):
                if signs[j]["name"] not in [
                    "condition",
                    "time",
                    "chat",
                    "support",
                    "positions",
                ]:
                    truefirst = False
            if not truefirst:
                if s["name"] in ["swap", "long", "short", "close"] and p[
                    "name"
                ] not in ["condition", "time", "chat", "support", "positions"]:
                    if "inputToken" not in s["args"] or s["args"]["inputToken"] == "":
                        s["args"]["inputToken"] = "outputToken"
                    # if "inputAmount" not in s["args"] or s["args"]["inputAmount"] == "":
                    # s["args"]["inputAmount"] = "outputAmount"
                if s["name"] in [
                    "transfer",
                    "bridge",
                    "deposit",
                    "withdraw",
                    "borrow",
                    "lend",
                    "repay",
                    "stake",
                    "unstake",
                    "lock",
                    "unlock",
                ] and p["name"] not in [
                    "condition",
                    "time",
                    "chat",
                    "support",
                    "positions",
                ]:
                    if "token" not in s["args"] or s["args"]["token"] == "":
                        s["args"]["token"] = "outputToken"
                    if "amount" not in s["args"] or s["args"]["amount"] == "":
                        s["args"]["amount"] = "outputAmount"
        processed.append(s)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_same_token_swap_perp(signs) -> list[dict[str, Any]]:
    """
    Remove same token swaps, remove input token on same token perps
    """
    processed = [signs[0]]
    skipped = 0
    for s in signs[1:]:
        if (
            s["name"] in ["swap"]
            and "inputToken" in s["args"]
            and "outputToken" in s["args"]
            and s["args"]["inputToken"] == s["args"]["outputToken"]
        ):
            skipped += 1
        else:
            processed.append(s)
    assert len(signs) == len(processed) + skipped, (signs, processed, skipped)
    signs = processed

    processed = []
    for s in signs:
        if (
            s["name"] in ["long", "short", "close"]
            and "inputToken" in s["args"]
            and "outputToken" in s["args"]
            and s["args"]["inputToken"] == s["args"]["outputToken"]
        ):
            del s["args"]["inputToken"]
        processed.append(s)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs

def process_same_chain_bridge(signs) -> list[dict[str, Any]]:
    """
    Remove same chain bridges
    """
    processed = []
    skipped = 0
    for s in signs:
        if (
            s["name"] in ["bridge"]
            and "sourceChainName" in s["args"]
            and "destinationChainName" in s["args"]
            and s["args"]["sourceChainName"] == s["args"]["destinationChainName"]
        ):
            skipped += 1
        else:
            processed.append(s)
    assert len(signs) == len(processed) + skipped, (signs, processed, skipped)
    signs = processed
    return signs


def process_repeats(signs) -> list[dict[str, Any]]:
    """
    Remove repeats
    """
    processed = [signs[0]]
    skipped = 0
    for s in signs[1:]:
        if s != processed[-1]:
            processed.append(s)
        else:
            skipped += 1
    assert len(signs) == len(processed) + skipped, (signs, processed, skipped)
    signs = processed
    return signs


def process_time(signs, message) -> list[dict[str, Any]]:
    """
    Edit arguments of a time condition to fix common AI mistakes
    """
    processed = []
    for ix in range(len(signs)):
        if ix <= len(signs) // 2:
            if (
                "tomorrow" in message[: len(message) // 2]
                and "start_time" in signs[ix]["args"]
                and "tomorrow" not in signs[ix]["args"]["start_time"]
            ):
                if signs[ix]["args"]["start_time"] != "now":
                    signs[ix]["args"]["start_time"] = (
                        signs[ix]["args"]["start_time"] + " tomorrow"
                    )
                if "recurrence" in signs[ix]["args"] and not any(
                    element in message.lower()
                    for element in ["every", "each", "recurrence"]
                ):
                    del signs[ix]["args"]["recurrence"]
            if (
                "today" in message[: len(message) // 2]
                and "start_time" in signs[ix]["args"]
                and "today" not in signs[ix]["args"]["start_time"]
            ):
                signs[ix]["args"]["start_time"] = (
                    signs[ix]["args"]["start_time"] + " today"
                )
                if "recurrence" in signs[ix]["args"] and not any(
                    element in message.lower()
                    for element in ["every", "each", "recurrence"]
                ):
                    del signs[ix]["args"]["recurrence"]
        if ix >= len(signs) // 2:
            if (
                "tomorrow" in message[len(message) // 2 :]
                and "start_time" in signs[ix]["args"]
                and "tomorrow" not in signs[ix]["args"]["start_time"]
            ):
                if signs[ix]["args"]["start_time"] != "now":
                    signs[ix]["args"]["start_time"] = (
                        signs[ix]["args"]["start_time"] + " tomorrow"
                    )
                if "recurrence" in signs[ix]["args"] and not any(
                    element in message.lower()
                    for element in ["every", "each", "recurrence"]
                ):
                    del signs[ix]["args"]["recurrence"]
            if (
                "today" in message[len(message) // 2 :]
                and "start_time" in signs[ix]["args"]
                and "today" not in signs[ix]["args"]["start_time"]
            ):
                signs[ix]["args"]["start_time"] = (
                    signs[ix]["args"]["start_time"] + " today"
                )
                if "recurrence" in signs[ix]["args"] and not any(
                    element in message.lower()
                    for element in ["every", "each", "recurrence"]
                ):
                    del signs[ix]["args"]["recurrence"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_duplicate_chain_name(signs) -> list[dict[str, Any]]:
    """
    Delete duplicate chain name
    """
    processed = []
    for ix in range(len(signs) - 1):
        if (
            "destinationChainName" in signs[ix + 1]["args"]
            and "chainName" in signs[ix]["args"]
        ):
            if (
                signs[ix]["args"]["chainName"]
                == signs[ix + 1]["args"]["destinationChainName"]
            ):
                del signs[ix]["args"]["chainName"]
        processed.append(signs[ix])
    processed.append(signs[-1])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_all_after_amount(signs) -> list[dict[str, Any]]:
    """
    Can't use an amount of a token for an action if the previous action is the same thing but used all of the token
    """
    processed = [signs[0]]
    count = 0
    for ix in range(1, len(signs)):
        if signs[ix]["name"] == signs[ix - 1]["name"]:
            if (
                "amount" in signs[ix]["args"]
                and "token" in signs[ix]["args"]
                and "amount" in signs[ix - 1]["args"]
                and "token" in signs[ix - 1]["args"]
            ):
                if (
                    signs[ix]["args"]["amount"] == signs[ix - 1]["args"]["amount"]
                    and signs[ix]["args"]["amount"] == "all"
                ):
                    if signs[ix]["args"]["token"] == signs[ix - 1]["args"]["token"]:
                        count += 1
                        processed.pop()
        processed.append(signs[ix])
    assert len(signs) == len(processed) + count, (signs, processed)
    signs = processed
    return signs


def process_chain_transfer_mistake(signs, chains) -> list[dict[str, Any]]:
    """
    Replace a transfer to a chain with actual bridge to the protocol
    """
    processed = []
    for s in signs:
        if s["name"] == "transfer" and "recipient" in s["args"]:
            x = (
                s["args"]["recipient"]
                .replace("_chain", "")
                .replace("chain", "")
                .replace("_protocol", "")
                .replace("protocol", "")
                .replace("_address", "")
                .replace("address", "")
                .replace(" ", "")
            )
            if x in chains:
                new = s
                new["name"] = "bridge"
                del new["args"]["recipient"]
                if "destinationChainName" not in new["args"]:
                    new["args"]["destinationChainName"] = x
                processed.append(new)
                continue
        processed.append(s)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_protocol_transfer_mistake(signs, protocols) -> list[dict[str, Any]]:
    """
    Replace a transfer to a protocol with actual usage of the protocol in the previous call and delete current
    """
    processed: list[Any] = []
    count = 0
    for ix, s0 in enumerate(signs):
        if (
            ix != 0
            and "protocolName" not in signs[ix - 1]["args"]
            and signs[ix - 1]["name"]
            not in ["condition", "time", "chat", "support", "positions"]
        ):
            if s0["name"] == "transfer" and "recipient" in s0["args"]:
                x = (
                    s0["args"]["recipient"]
                    .replace("_protocol", "")
                    .replace("protocol", "")
                    .replace("_address", "")
                    .replace("address", "")
                )
                if x in protocols:
                    processed[-1]["args"]["protocolName"] = x
                    count += 1
                    continue
        processed.append(s0)
    assert len(signs) == len(processed) + count, (signs, processed)
    signs = processed
    return signs


def process_price_condition_2(signs) -> list[dict[str, Any]]:
    """
    Update input/output token price condition intelligently
    Delete "side" param since its only used to be helpful for previous logic
    """
    processed = []
    for ix in range(len(signs)):
        s0 = signs[ix]
        if (
            ix > 0
            and "subject" in s0["args"]
            and s0["args"]["subject"].lower() in ["outputtoken", "outputamount", "usd"]
        ):
            if "side" in signs[ix - 1]["args"]:
                if (
                    signs[ix - 1]["args"]["side"] == "sell"
                    and "inputToken" in signs[ix - 1]["args"]
                ):
                    s0["args"]["subject"] = signs[ix - 1]["args"]["inputToken"]
                if (
                    signs[ix - 1]["args"]["side"] == "buy"
                    and "outputToken" in signs[ix - 1]["args"]
                ):
                    s0["args"]["subject"] = signs[ix - 1]["args"]["outputToken"]
            else:
                if (
                    "inputAmount" in signs[ix - 1]["args"]
                    and "inputToken" in signs[ix - 1]["args"]
                ):
                    if (
                        signs[ix - 1]["args"]["inputToken"] in ["usdc", "usdt", "dai"]
                        and s0["args"]["type"] == "price"
                        and (
                            float(s0["args"]["value"]) < 0.89
                            or float(s0["args"]["value"]) > 1.11
                        )
                        and "outputToken" in signs[ix - 1]["args"]
                    ):
                        s0["args"]["subject"] = signs[ix - 1]["args"]["outputToken"]
                    else:
                        s0["args"]["subject"] = signs[ix - 1]["args"]["inputToken"]
                elif (
                    "outputAmount" in signs[ix - 1]["args"]
                    and "outputToken" in signs[ix - 1]["args"]
                ):
                    s0["args"]["subject"] = signs[ix - 1]["args"]["outputToken"]
                elif "outputToken" in signs[ix - 1]["args"]:
                    s0["args"]["subject"] = signs[ix - 1]["args"]["outputToken"]
                elif "token" in signs[ix - 1]["args"]:
                    s0["args"]["subject"] = signs[ix - 1]["args"]["token"]
                elif "inputToken" in signs[ix - 1]["args"]:
                    s0["args"]["subject"] = signs[ix - 1]["args"]["inputToken"]
        if (
            len(signs) > 1
            and ix == 0
            and "subject" in s0["args"]
            and s0["args"]["subject"].lower() in ["outputtoken", "outputamount", "usd"]
        ):
            if "side" in signs[ix + 1]["args"]:
                if signs[ix + 1]["args"]["side"] == "sell":
                    s0["args"]["subject"] = signs[ix + 1]["args"]["inputToken"]
                if signs[ix + 1]["args"]["side"] == "buy":
                    s0["args"]["subject"] = signs[ix + 1]["args"]["outputToken"]
            else:
                if (
                    "inputAmount" in signs[ix + 1]["args"]
                    and "inputToken" in signs[ix + 1]["args"]
                ):
                    if (
                        signs[ix + 1]["args"]["inputToken"] in ["usdc", "usdt", "dai"]
                        and s0["args"]["type"] == "price"
                        and (
                            float(s0["args"]["value"]) < 0.89
                            or float(s0["args"]["value"]) > 1.11
                        )
                        and "outputToken" in signs[ix + 1]["args"]
                    ):
                        s0["args"]["subject"] = signs[ix + 1]["args"]["outputToken"]
                    else:
                        s0["args"]["subject"] = signs[ix + 1]["args"]["inputToken"]
                elif (
                    "outputAmount" in signs[ix + 1]["args"]
                    and "outputToken" in signs[ix + 1]["args"]
                ):
                    s0["args"]["subject"] = signs[ix + 1]["args"]["outputToken"]
                elif "inputToken" in signs[ix + 1]["args"]:
                    s0["args"]["subject"] = signs[ix + 1]["args"]["inputToken"]
                elif "token" in signs[ix + 1]["args"]:
                    s0["args"]["subject"] = signs[ix + 1]["args"]["token"]
                elif "outputToken" in signs[ix + 1]["args"]:
                    s0["args"]["subject"] = signs[ix + 1]["args"]["outputToken"]
        for iy in range(ix + 1, len(signs)):
            if "operator" in signs[iy]["args"]:
                if "subject" in signs[iy]["args"] and signs[iy]["args"][
                    "subject"
                ].lower() in ["outputtoken", "outputamount"]:
                    ss0 = signs[iy]
                    ss0["args"]["subject"] = s0["args"]["subject"]
            else:
                break
        processed.append(s0)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    processed = []
    for ix, s0 in enumerate(signs):
        if "side" in s0["args"]:
            del s0["args"]["side"]
        processed.append(s0)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_price_condition(signs):
    """
    Fix mistaken input/output token price condition intelligently
    """
    processed = []
    for ix in range(len(signs)):
        s0 = signs[ix]
        if ix > 0 and "subject" in s0["args"]:
            if (
                "outputToken" in signs[ix - 1]["args"]
                and "inputToken" in signs[ix - 1]["args"]
            ):
                if (
                    s0["args"]["subject"] in ["usdc", "usdt", "dai"]
                    and s0["args"]["type"] == "price"
                    and (
                        float(s0["args"]["value"]) < 0.89
                        or float(s0["args"]["value"]) > 1.11
                    )
                ):
                    if s0["args"]["subject"] == signs[ix - 1]["args"]["outputToken"]:
                        s0["args"]["subject"] = signs[ix - 1]["args"]["inputToken"]
                    elif s0["args"]["subject"] == signs[ix - 1]["args"]["inputToken"]:
                        s0["args"]["subject"] = signs[ix - 1]["args"]["outputToken"]
        if len(signs) > 1 and ix == 0 and "subject" in s0["args"]:
            if (
                "outputToken" in signs[ix + 1]["args"]
                and "inputToken" in signs[ix + 1]["args"]
            ):
                if (
                    s0["args"]["subject"] in ["usdc", "usdt", "dai"]
                    and s0["args"]["type"] == "price"
                    and (
                        float(s0["args"]["value"]) < 0.89
                        or float(s0["args"]["value"]) > 1.11
                    )
                ):
                    if s0["args"]["subject"] == signs[ix + 1]["args"]["outputToken"]:
                        s0["args"]["subject"] = signs[ix + 1]["args"]["inputToken"]
                    elif s0["args"]["subject"] == signs[ix + 1]["args"]["inputToken"]:
                        s0["args"]["subject"] = signs[ix + 1]["args"]["outputToken"]
        processed.append(s0)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_token_amount_assignment(signs, message: str):
    """
    Assign token amounts properly based on ordering in actual user message
    """
    # to input/output token for swaps
    processed = []
    for ix in range(len(signs)):
        s0 = signs[ix]
        if "swap" in s0["name"] and "outputToken" in s0["args"]:
            target = s0["args"]["outputToken"]
            msglist0 = message.replace("'", "").split(" ")
            idcs = [i - 1 for i, x in enumerate(msglist0) if x == target]
            for idc in idcs:
                try:
                    y = float(
                        msglist0[idc].replace("%", "").replace("$", "").replace(",", "")
                    )
                    if (
                        (
                            "all" not in message
                            and "everything" not in message
                            and "inputAmount" in s0["args"]
                            and s0["args"]["inputAmount"] == "all"
                        )
                        or (
                            "inputAmount" in s0["args"]
                            and y == float(s0["args"]["inputAmount"])
                        )
                        or (
                            "inputAmount" not in s0["args"]
                            and "outputAmount" not in s0["args"]
                        )
                    ):
                        s0["args"]["outputAmount"] = msglist0[idc]
                        if "inputAmount" in s0["args"]:
                            del s0["args"]["inputAmount"]
                    break
                except Exception:
                    # print(e)
                    pass
        processed.append(s0)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    # to inputToken and inputAmount for swaps
    processed = []
    names = [signs[ix]["name"] for ix in range(len(signs))]
    for ix in range(len(signs)):
        s0 = signs[ix]
        if (
            "swap" in s0["name"]
            and "inputToken" in s0["args"]
            and ("inputAmount" not in s0["args"])
            and "outputAmount" not in s0["args"]
        ):
            target = s0["args"]["inputToken"]
            msglist0 = [x for x in message.split(" ") if x != "of"]
            idcs0 = [[i - 2, i - 1] for i, x in enumerate(msglist0) if x == target]
            for idc0 in idcs0:
                doubleprev = (
                    msglist0[idc0[0]].replace("%", "").replace("$", "").replace(",", "")
                )
                prev = (
                    msglist0[idc0[1]].replace("%", "").replace("$", "").replace(",", "")
                )
                if prev == "all":
                    s0["args"]["inputAmount"] = "all"
                    break
                if prev == "half":
                    s0["args"]["inputAmount"] = "half"
                    break
                try:
                    if doubleprev not in names:
                        y = float(prev)
                        s0["args"]["inputAmount"] = msglist0[idc0[1]]
                        break
                    elif doubleprev == s0["name"]:
                        y = float(prev)
                        s0["args"]["inputAmount"] = msglist0[idc0[1]]
                        break
                except Exception:
                    # print(e)
                    pass
        processed.append(s0)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    # to token and amount for other actions
    processed = []
    for ix in range(len(signs)):
        s0 = signs[ix]
        if "token" in s0["args"] and ("amount" not in s0["args"]):
            target = s0["args"]["token"]
            msglist0 = [x for x in message.split(" ") if x != "of"]
            idcs = [i - 1 for i, x in enumerate(msglist0) if x == target]
            for idc in idcs:
                prev = msglist0[idc].replace("%", "").replace("$", "").replace(",", "")
                if prev == "all":
                    s0["args"]["amount"] = "all"
                    break
                if prev == "half":
                    s0["args"]["amount"] = "half"
                    break
                try:
                    y = float(prev)
                    s0["args"]["amount"] = msglist0[idc]
                    break
                except Exception:
                    # print(e)
                    pass
        processed.append(s0)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed

    # remove inputAmount if all and outputAmount
    processed = []
    for ix in range(len(signs)):
        if (
            "swap" in signs[ix]["name"]
            and "inputAmount" in signs[ix]["args"]
            and "outputAmount" in signs[ix]["args"]
            and signs[ix]["args"]["inputAmount"] == "all"
        ):
            del signs[ix]["args"]["inputAmount"]
        if (
            "swap" in signs[ix]["name"]
            and "inputToken" in signs[ix]["args"]
            and "outputToken" in signs[ix]["args"]
            and "outputAmount" in signs[ix]["args"]
            and signs[ix]["args"]["inputToken"] == signs[ix]["args"]["outputToken"]
        ):
            del signs[ix]["args"]["inputToken"]
        processed.append(signs[ix])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_nested_conditions(signs) -> list[dict[str, Any]]:
    """
    Delete conditional parameters in mistakenly added to other calls
    """
    processed = []
    for ix, s0 in enumerate(signs):
        if (
            ix == 0
            and "operator" in s0["args"]
            and len(signs) > 1
            and signs[ix + 1]["name"] not in ["condition", "time"]
        ):
            del s0["args"]["operator"]
        if (
            ix == len(signs) - 1
            and "operator" in s0["args"]
            and len(signs) > 1
            and signs[ix - 1]["name"] not in ["condition", "time"]
        ):
            del s0["args"]["operator"]
        if (
            ix != 0
            and ix != len(signs) - 1
            and "operator" in s0["args"]
            and len(signs) > 2
            and signs[ix + 1]["name"] not in ["condition", "time"]
            and signs[ix - 1]["name"] not in ["condition", "time"]
        ):
            del s0["args"]["operator"]
        processed.append(s0)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_empty_protocol_names_backwards(signs) -> list[dict[str, Any]]:
    """
    Assign protocol names to other calls in sequence if needed but not there based on one call (backwards)
    """
    processed = []
    first = -1
    for f0 in range(len(signs)):
        if signs[f0]["name"] not in ["condition", "time"]:
            first = f0
            break
        processed.append(signs[f0])
    if first != -1:
        x = signs[first]
        if x["name"] in [
            "claim",
            "unstake",
            "unlock",
            "vote",
            "repay",
            "close",
        ] and ("protocolName" not in x["args"] or x["args"]["protocolName"] == ""):
            x["args"]["protocolName"] = "all"
        processed.append(x)
        for s in range(first + 1, len(signs)):
            cur = signs[s]
            if cur["name"] not in [
                "swap",
                "bridge",
                "transfer",
                "condition",
                "time",
            ] and (
                "protocolName" not in cur["args"] or cur["args"]["protocolName"] == ""
            ):
                for ss in range(s - 1, -1, -1):
                    if (
                        signs[ss]["name"]
                        not in ["swap", "bridge", "transfer", "condition", "time"]
                        and "protocolName" in signs[ss]["args"]
                    ):
                        cur["args"]["protocolName"] = signs[ss]["args"]["protocolName"]
                        break
            processed.append(cur)
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def process_empty_protocol_names_forwards(signs) -> list[dict[str, Any]]:
    """
    Assign protocol names to other calls in sequence if needed but not there based on one call (forwards)
    """
    processed = []
    for s in range(len(signs) - 1):
        cur = signs[s]
        nex = signs[s + 1]
        # print(cur, nex)
        if (
            (cur["name"] == "swap" or cur["name"] == "bridge")
            and "protocolName" in cur["args"]
            and "protocolName" in nex["args"]
            and nex["name"] not in ["swap", "bridge"]
            and cur["args"]["protocolName"] == nex["args"]["protocolName"]
        ):
            if not (
                s > 0
                and signs[s - 1]["name"] in ["swap", "bridge"]
                and "protocolName" in signs[s - 1]["args"]
                and signs[s - 1]["args"]["protocolName"] == cur["args"]["protocolName"]
            ):
                del cur["args"]["protocolName"]
        # if "poolName" in cur["args"] and "protocolName" not in cur["args"]:
        # del cur["args"]["poolName"]
        prevnames0 = []
        for ss in range(s):
            prevnames0.append(signs[ss]["name"])
        prevnames = set(prevnames0)
        if (
            (
                prevnames == {"condition", "time"}
                or prevnames == {"time", "condition"}
                or prevnames == {"time"}
                or prevnames == {"condition"}
                or s == 0
            )
            and cur["name"] not in ["swap", "bridge", "condition", "time", "transfer"]
            and ("protocolName" not in cur["args"] or cur["args"]["protocolName"] == "")
        ):
            for ss in range(s, len(signs) - 1):
                if "protocolName" in signs[ss]["args"]:
                    cur["args"]["protocolName"] = signs[ss]["args"]["protocolName"]
        processed.append(cur)
    processed.append(signs[-1])
    assert len(signs) == len(processed), (signs, processed)
    signs = processed
    return signs


def remove_first_instance(sentence, word):
    # Regular expression to match the word with potential preceding and following spaces or commas
    pattern = r"(\s*,?\s*)\b{}\b".format(re.escape(word))

    # Function to replace the match considering different scenarios
    def replace_func(match):
        # If there's a comma before the word, remove the comma along with the word
        if match.group(1).startswith(","):
            return ""
        # Otherwise, just remove the word
        else:
            return match.group(1)

    # Replace the first instance of the pattern using the custom function
    return re.sub(pattern, replace_func, sentence, count=1)


def processing(
    results,
    response,
    message: str,
    updated,
    chains,
    protocols,
    tokens,
    shortened_chains,
    rxtk,
    rxtk2,
):
    signs: list[dict[str, Any]] = []
    for ix, data in enumerate(results):
        try:
            name = data.name
        except Exception:
            # print(e)
            name = data["name"]
        try:
            args = to_lowercase(json.loads(data.arguments))
        except Exception:
            # print(e)
            args = to_lowercase(json.loads(data["arguments"]))
        if name == "time" and args == {"start_time": "now"}:
            continue
        if ix < len(results) - 1 and name == "bridge":
            data2 = results[ix + 1]
            try:
                name2 = data2.name
            except Exception:
                # print(e)
                name2 = data2["name"]
            try:
                args2 = to_lowercase(json.loads(data2.arguments))
            except Exception:
                # print(e)
                args2 = to_lowercase(json.loads(data2["arguments"]))
            args2.pop("protocolName", None)
            if name == name2 and args == args2:
                if response and isinstance(response, str):
                    response = remove_first_instance(response, name)
                elif response:
                    try:
                        response.content = remove_first_instance(response.content, name)
                    except Exception as e:
                        print(e)
                        pass
                continue
        if ix < len(results) - 1 and name == "time" and "recurrence" in args:
            data2 = results[ix + 1]
            try:
                name2 = data2.name
            except Exception:
                # print(e)
                name2 = data2["name"]
            try:
                args2 = to_lowercase(json.loads(data2.arguments))
            except Exception:
                # print(e)
                args2 = to_lowercase(json.loads(data2["arguments"]))
            args2["recurrence"] = args["recurrence"]
            if name == name2 and args == args2:
                if response and isinstance(response, str):
                    response = remove_first_instance(response, name)
                elif response:
                    try:
                        response.content = remove_first_instance(response.content, name)
                    except Exception as e:
                        print(e)
                        pass
                continue
        args_basic_process_early(name, args, message, tokens)
        args_token0(args, updated)
        args_token1(args, updated)
        args_tokens_pools(args, updated, message)
        args_tokens_protocols_chains(
            name, args, chains, message, updated, protocols, shortened_chains
        )
        args_token2(args, message, updated)
        args_protocols_pools(args, message, updated, protocols, name, tokens)
        args_chain(args, message, updated, chains, shortened_chains)
        args_amount0(args, message, tokens)
        args_amount1(args, message, tokens)
        args_amount_units2(args)
        args_amount2(args)
        args_amount3(args)
        args_basic_process_late(
            name, args, signs, chains, tokens, shortened_chains, message
        )
        args_condition_args(args)
        args_value_slip(args, name, protocols, tokens, message)
        args_subj(args, rxtk, rxtk2, chains)
        args_time(args)
        addcond = args_cond(name, args)
        addtime = args_time_cond(name, args)
        x = {"name": name, "args": args}
        if len(signs) > 0 and x == signs[-1]:
            continue
        signs.append(x)
        if (
            addcond != {}
            and isinstance(addcond["args"], dict)
            and list(addcond["args"].keys()) != ["operator"]
        ):
            signs.append(addcond)
        if addtime != {} and addtime != {"name": "time", "args": {"start_time": "now"}}:
            signs.append(addtime)
    return signs, response


def args_time_cond(name: str, args):
    """
    Add time call if currently nested
    """
    addtime: dict[str, Any] = {}
    if name != "time" and (
        "start_time" in args or "recurrence" in args or "end_time" in args
    ):
        addtime = {"name": "time", "args": {}}
        if "operator" in args:
            addtime["args"]["operator"] = args["operator"]
            del args["operator"]
        if "start_time" in args:
            addtime["args"]["start_time"] = args["start_time"]
            del args["start_time"]
        if "end_time" in args:
            if (
                "start_time" in args and args["start_time"] != args["end_time"]
            ) or "start_time" not in args:
                addtime["args"]["end_time"] = args["end_time"]
            del args["end_time"]
        if "recurrence" in args:
            if args["recurrence"] is not None:
                addtime["args"]["recurrence"] = args["recurrence"]
            del args["recurrence"]
    elif "time" in args:
        addtime = {"name": "time", "args": args["time"]}
        del args["time"]
    return addtime


def args_cond(name: str, args):
    """
    Add conditional call if currently nested
    """
    addcond: dict[str, Any] = {}
    if name != "condition" and (
        "subject" in args
        or "comparator" in args
        or "value" in args
        or "value_token" in args
        or "type" in args
        or "period" in args
    ):
        addcond = args_cond_0(args)
        if "operator" in args:
            del args["operator"]
        if "subject" in args:
            del args["subject"]
        if "comparator" in args:
            del args["comparator"]
        if "value" in args:
            del args["value"]
        if "value_token" in args:
            del args["value_token"]
        if "period" in args:
            del args["period"]
        if "start_time" in args:
            del args["start_time"]
        if "recurrence" in args:
            del args["recurrence"]
        if "end_time" in args:
            del args["end_time"]
        if "type" in args:
            del args["type"]
        if "value_token" in addcond and (
            addcond["value_token"] == "dollars" or addcond["value_token"] == "usd"
        ):
            del addcond["args"]["value_token"]
    elif "condition" in args:
        addcond = {"name": "condition", "args": args["condition"]}
        del args["condition"]
    return addcond


def args_cond_0(args) -> dict[str, Any]:
    """
    Add conditional call if currently nested
    """
    addcond: dict[str, Any]
    if "subject" in args and args["subject"] == "time":
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
        if "operator" in args:
            addcond["args"]["operator"] = args["operator"]
        if "subject" in args:
            addcond["args"]["subject"] = args["subject"]
        if "comparator" in args:
            addcond["args"]["comparator"] = args["comparator"]
        if "value" in args:
            addcond["args"]["value"] = args["value"]
        if "value_token" in args:
            addcond["args"]["value_token"] = args["value_token"]
        if "type" in args:
            addcond["args"]["type"] = args["type"]
        if "period" in args:
            addcond["args"]["period"] = args["period"]
    return addcond


def args_time(args) -> None:
    """
    Clean up time calls
    """
    if (
        "start_time" in args
        and "end_time" in args
        and args["end_time"] == args["start_time"]
    ):
        del args["end_time"]
    if "start_time" in args and "end_time" in args and args["end_time"] == "tomorrow":
        if args["start_time"] != "now":
            args["start_time"] = args["start_time"] + " tomorrow"
        del args["end_time"]
    if "recurrence" in args and args["recurrence"] is None:
        del args["recurrence"]
    if "start_time" in args and args["start_time"] == "this time":
        args["start_time"] = "now"


def args_subj(args, rxtk, rxtk2, chains) -> None:
    """
    Clean up conditional arguments and label type
    """
    if "subject" in args:
        settype = False
        if "type" not in args or args["type"] not in [
            "price",
            "market cap",
            "balance",
            "gas",
            "yield",
            "ltv",
            "fdv",
            "health factor",
            "funding rate",
            "open interest",
        ]:
            settype = True
        if args["subject"] == "gwei":
            args["subject"] = "gas"
        if (
            args["subject"].lower() in chains
            and "type" in args
            and args["type"] == "gas"
        ):
            args["subject"] = "gas"
        if args["subject"] == "borrow":
            args["subject"] = "borrow apy"
        if "type" in args and args["type"] == "usd":
            args["value_units"] = "usd"
            args["type"] = "price"
        args_subj_0(args, settype, rxtk, rxtk2)
        if "rewards" in args["subject"] and args["type"] == "price":
            if settype:
                args["type"] = "balance"
        if args["subject"] == "":
            args["subject"] = "outputToken"
        if (
            "health factor" in args["subject"]
            and args["subject"].replace("health factor", "").strip() != ""
        ):
            args["subject"] = args["subject"].replace("health factor", "").strip()
            args["type"] = "health factor"
        if args["subject"] == "marketcap":
            args["type"] = "market cap"
            args["subject"] = "outputToken"
        if args["subject"] == "fdv":
            if "value" in args:
                if args["type"] == "thousand":
                    args["value"] = str(float(args["value"]) * 1000)
                if args["type"] == "million":
                    args["value"] = str(float(args["value"]) * 1000000)
                if args["type"] == "billion":
                    args["value"] = str(float(args["value"]) * 1000000000)
                if args["type"] == "trillion":
                    args["value"] = str(float(args["value"]) * 1000000000000)
            args["type"] = "fdv"
            args["subject"] = "outputToken"
        if args["subject"] in ["funding rate", "funding"]:
            args["subject"] = "outputToken"
            args["type"] = "funding rate"
        if args["subject"] in ["open interest"]:
            args["subject"] = "outputToken"
            args["type"] = "open interest"


def args_subj_0(args, settype, rxtk, rxtk2) -> None:
    """
    Clean up conditional arguments and label type
    """
    if args["subject"]:
        args["subject"] = args["subject"].replace(" trades", "")
    if "price" in args["subject"]:
        args["subject"] = args["subject"].replace(" price", "")
        args["subject"] = args["subject"].replace("_market_price", "")
        args["subject"] = args["subject"].replace("_price", "")
        args["subject"] = args["subject"].replace("price_of_", "")
        args["subject"] = args["subject"].replace("price", "")
        if args["subject"] == "":
            args["subject"] = "outputToken"
        if settype:
            args["type"] = "price"
    elif "balance" in args["subject"]:
        args["subject"] = args["subject"].replace(" balance", "")
        args["subject"] = args["subject"].replace("_balance", "")
        args["subject"] = args["subject"].replace("balance_of_", "")
        args["subject"] = args["subject"].replace("balance", "")
        if args["subject"] == "":
            args["subject"] = "outputToken"
        if settype:
            args["type"] = "balance"
    elif (
        "market cap" in args["subject"]
        or "market_cap" in args["subject"]
        or ("type" in args and "market_cap" in args["type"])
        or ("type" in args and "marketcap" in args["type"])
    ):
        args["subject"] = args["subject"].replace(" market cap", "")
        args["subject"] = args["subject"].replace("_market cap", "")
        args["subject"] = args["subject"].replace("_market_cap", "")
        args["subject"] = args["subject"].replace("market_cap_of_", "")
        args["subject"] = args["subject"].replace("market cap", "")
        args["subject"] = args["subject"].replace("market_cap", "")
        if args["subject"] == "":
            args["subject"] = "outputToken"
        if settype:
            args["type"] = "market cap"
    elif "gas" in args["subject"]:
        if settype:
            args["type"] = "gas"
    elif "apy" in args["subject"]:
        if settype:
            args["type"] = "yield"
    elif "ltv" in args["subject"]:
        if settype:
            args["type"] = "ltv"
    else:
        if settype:
            args["type"] = "price"
    if args["subject"]:
        args["subject"] = re.sub(
            rxtk + "_" + rxtk2, r"\g<tkn>/\g<tkn2>", args["subject"]
        )


def args_value_slip(args, name, protocols, tokens, message: str):
    """
    Clean up conditional arguments more
    """
    if "value" in args:
        if "price" in args["value"]:
            args["value"] = args["value"].replace(" price ", "")
            args["value"] = args["value"].replace("price ", "")
            args["value"] = args["value"].replace("_price", "")
            args["value"] = args["value"].replace("price_of_", "")
            args["value"] = args["value"].replace("price", "")
            args["type"] = "price"
        elif "market cap" in args["value"] or "market_cap" in args["value"]:
            args["value"] = args["value"].replace(" market cap", "")
            args["value"] = args["value"].replace("_market cap", "")
            args["value"] = args["value"].replace("_market_cap", "")
            args["value"] = args["value"].replace("market_cap_of_", "")
            args["value"] = args["value"].replace("market cap", "")
            args["value"] = args["value"].replace("market_cap", "")
            args["type"] = "market cap"
        split = [xx for xx in args["value"].split(" ") if xx != ""]
        if len(split) == 2:
            if not (split[0] in protocols and split[1] in tokens):
                args["value"] = split[0]
                args["value_token"] = split[1]
                if args["value_token"] == "dollars" or args["value_token"] == "usd":
                    del args["value_token"]
        if "value_units" in args and args["value_units"] == "%":
            if args["value"][-1] != "%":
                args["value"] = args["value"] + "%"
            del args["value_units"]
            if "type" in args and args["type"] == "apy":
                args["subject"] = "apy"
                args["type"] = "yield"
        if "value_units" in args and args["value_units"] == "cents" and "value" in args:
            try:
                y = float(args["value"])
                args["value"] = str(y / 100)
                args["value_units"] = "usd"
            except Exception:
                # print(e)
                pass
        if (
            "value_units" in args
            and args["value_units"] not in ["gwei", "usd"]
            and "value_token" not in args
        ):
            args["value_token"] = args["value_units"]
            del args["value_units"]
        if "value_units" in args and "value_token" in args:
            args["value_token"] = args["value_units"]
            del args["value_units"]
        if "valueType" in args and args["valueType"] == "apy":
            if "type" in args and args["type"] == "borrow":
                args["subject"] = "borrow apy"
            else:
                args["subject"] = "apy"
            args["type"] = "yield"

    if "slippage" in args and (name != "swap" or "slippage" not in message.lower()):
        del args["slippage"]


def args_condition_args(args) -> None:
    """
    Clean up conditional arguments
    """
    if "comparator" in args:
        unders = [
            "below",
            "is under",
            "lt",
            "lte",
            "less than",
            "is less than",
            "less_than",
            "lessThan",
            "lessthan",
            "<",
            "goes below",
            "sub",
            "is at or below",
        ]
        overs = [
            "above",
            "is over",
            "gt",
            "gte",
            "greater than",
            "is greater than",
            "greater_than",
            "greaterThan",
            "greaterthan",
            ">",
            "goes above",
            "is at or above",
            "exceeds",
            "surpasses",
        ]
        equals = ["equals", "hit", "hits", "=", "is at", "reaches", "near", "goes to"]
        notequals = ["notequals"]
        if args["comparator"] in unders:
            args["comparator"] = "<="
        if args["comparator"] in overs:
            args["comparator"] = ">="
        if (
            args["comparator"] in equals
            or args["comparator"] == "is"
            or args["comparator"] == "at"
        ):
            args["comparator"] = "=="
        if args["comparator"] in notequals:
            args["comparator"] = "!="
        if args["comparator"] in [
            "dips",
            "decrease",
            "decreases",
            "decrease by",
            "decreases by",
        ]:
            args["comparator"] = "<="
            if "value" in args and args["value"][0] != "-":
                args["value"] = "-" + args["value"]
        if (
            "value" not in args
            and args["comparator"]
            .replace("<=", "")
            .replace(">=", "")
            .replace("==", "")
            .replace("!=", "")
            != ""
        ):
            args["value"] = (
                args["comparator"]
                .replace("<=", "")
                .replace(">=", "")
                .replace("==", "")
                .replace("!=", "")
            )
            if "<=" in args["comparator"]:
                args["comparator"] = "<="
            elif ">=" in args["comparator"]:
                args["comparator"] = ">="
            elif "==" in args["comparator"]:
                args["comparator"] = "=="
            elif "!=" in args["comparator"]:
                args["comparator"] = "!="
    if "period" in args:
        elms = [
            "day",
            "days",
            "month",
            "months",
            "year",
            "years",
            "hour",
            "hours",
            "minute",
            "minutes",
            "second",
            "seconds",
            "forever",
        ]
        if not any(element in args["period"] for element in elms):
            del args["period"]
        else:
            rxnm = r"(?P<num>\d+\.\d*|\d+|\.\d+)"
            rxelm = r"(?P<elm>" + r"|".join(elms) + r")"
            args["period"] = re.sub(
                rxnm + r"\s*" + rxelm, r"\g<num> \g<elm>", args["period"].lower()
            )


def args_basic_process_late(
    name: str, args, signs, chains, tokens, shortened_chains, message
):
    """
    Generally clean up arguments that need to be cleaned up after main edits
    """
    if (
        (name == "swap" or name == "bridge")
        and "protocolName" in args
        and len(signs) > 0
        and "protocolName" in signs[-1]["args"]
        and signs[-1]["name"] not in ["swap", "bridge"]
        and args["protocolName"] == signs[-1]["args"]["protocolName"]
    ):
        del args["protocolName"]
    if (
        "poolName" in args
        and "protocolName" in args
        and args["protocolName"].lower() == "all"
    ):
        del args["poolName"]
    if (
        "token" in args
        and isinstance(args["token"], str)
        and args["token"] in chains
        and args["token"] not in tokens
    ):
        if name != "bridge":
            if "chainName" not in args:
                args["chainName"] = args["token"]
            # long_chains = {value: key for key, value in shortened_chains.items()}
            # if args["token"] in long_chains and long_chains[args["token"]] in message.lower().split(" "):
            # args["token"] = long_chains[args["token"]]
            # else:
            # del args["token"]
            del args["token"]
        if name == "bridge":
            if "sourceChainName" not in args:
                args["sourceChainName"] = args["token"]
            # long_chains = {value: key for key, value in shortened_chains.items()}
            # if args["token"] in long_chains and long_chains[args["token"]] in message.lower().split(" "):
            # args["token"] = long_chains[args["token"]]
            # else:
            # del args["token"]
            del args["token"]
    if (
        "inputToken" in args
        and isinstance(args["inputToken"], str)
        and args["inputToken"] in chains
        and args["inputToken"] not in tokens
    ):
        if name != "bridge":
            if "chainName" not in args:
                args["chainName"] = args["inputToken"]
            del args["inputToken"]
        if name == "bridge":
            if "sourceChainName" not in args:
                args["sourceChainName"] = args["inputToken"]
            del args["inputToken"]
    if (
        name in ["long", "short"]
        and "inputToken" in args
        and ("outputToken" not in args or args["outputToken"] == "")
    ):
        args["outputToken"] = args["inputToken"]
    if (
        "token" in args
        and isinstance(args["token"], str)
        and "poolName" in args
        and args["poolName"] + " lp" == args["token"]
    ):
        del args["poolName"]
    if (
        "amount_units" in args
        and (
            "token" not in args
            or ("token" in args and args["token"] == "")
            or ("token" in args and args["token"] not in tokens)
        )
        and args["amount_units"].lower() in tokens
    ):
        args["token"] = args["amount_units"]
        del args["amount_units"]
    if (
        "sourceChainName" in args
        and "destinationChainName" in args
        and args["sourceChainName"].lower() == args["destinationChainName"].lower()
    ):
        del args["sourceChainName"]
    if (
        "amount_units" in args
        and "token" in args
        and args["token"].lower() == "outputtoken"
    ):
        del args["amount_units"]
    if "period" in args and "value" in args:
        args["period"] = args["period"].replace("token", "")
        args["period"] = args["period"].replace(" ", "")
        if args["period"] in tokens:
            args["value"] = args["value"] + " " + args["period"]
            del args["period"]
    if "inputAmountUnits" in args and "inputToken" in args:
        if args["inputToken"] == "outputToken":
            del args["inputAmountUnits"]
        try:
            if sorted(args["inputAmountUnits"]) == sorted(args["inputToken"]):
                del args["inputAmountUnits"]
        except Exception as e:
            print(e)
            pass
        if "outputToken" in args and args["inputToken"] == args["outputToken"]:
            args["inputToken"] = args["inputAmountUnits"]
            del args["inputAmountUnits"]
    if name == "condition" and "amount_units" in args:
        if args["amount_units"] != "":
            args["value_token"] = args["amount_units"]
            if args["value_token"] == "dollars" or args["value_token"] == "usd":
                del args["value_token"]
        del args["amount_units"]
    if name == "condition" and "token" in args:
        if args["token"] != "":
            args["value_token"] = args["token"]
            if args["value_token"] == "dollars" or args["value_token"] == "usd":
                del args["value_token"]
        del args["token"]
    if name == "condition" and "subject" in args and "value_token" in args:
        if args["value_token"] in args["subject"] and args["value_token"] != "usd":
            del args["value_token"]
    if name == "condition" and "subject" in args and "value_units" in args:
        if args["value_units"] in args["subject"] and args["value_units"] != "usd":
            del args["value_units"]
    if (
        "side" in args
        and args["side"] == "buy"
        and (
            "outputToken" not in args
            or ("outputToken" in args and args["outputToken"] == "")
        )
    ):
        if "inputToken" in args and "inputAmountUnits" in args:
            args["outputToken"] = args["inputToken"]
            args["inputToken"] = args["inputAmountUnits"]
            del args["inputAmountUnits"]
        elif "inputToken" in args:
            args["outputToken"] = args["inputToken"]
            del args["inputToken"]
        elif "inputAmountUnits" in args:
            args["outputToken"] = args["inputAmountUnits"]
            del args["inputAmountUnits"]


def args_amount3(args) -> None:
    """
    Clean up multi amount/token pairs
    """
    if (
        "amount" in args
        and "token" in args
        and isinstance(args["amount"], list)
        and isinstance(args["token"], list)
    ):
        assert len(args["amount"]) == len(args["token"])
        zipped = set(zip(args["amount"], args["token"]))
        newa = []
        newt = []
        for z in zipped:
            newa.append(z[0])
            newt.append(z[1])
        assert len(newa) == len(newt)
        if len(newa) > 1:
            args["amount"] = newa
            args["token"] = newt
        elif len(newa) == 1:
            args["amount"] = newa[0]
            args["token"] = newt[0]
        else:
            raise Exception(f"{args['amount']} {args['token']}")
    if (
        "amount" in args
        and "token" in args
        and isinstance(args["amount"], list)
        and isinstance(args["token"], str)
    ):
        args["amount"] = args["amount"][0]
    if (
        "amount" in args
        and "token" in args
        and isinstance(args["amount"], list)
        and isinstance(args["token"], list)
    ):
        assert len(args["amount"]) == len(args["token"])
        newt = []
        newa = []
        for t, a in zip(args["token"], args["amount"]):
            if t == a and t == "all":
                continue
            else:
                newt.append(t)
                newa.append(a)
        assert len(newa) == len(newt)
        if len(newa) > 1:
            args["amount"] = newa
            args["token"] = newt
        elif len(newa) == 1:
            args["amount"] = newa[0]
            args["token"] = newt[0]
        else:
            raise Exception(f"{args['amount']} {args['token']}")


def args_amount2(args) -> None:
    """
    Clean up multi amount/token pairs
    """
    if (
        "inputAmount" in args
        and "inputToken" in args
        and isinstance(args["inputAmount"], list)
        and isinstance(args["inputToken"], list)
    ):
        assert len(args["inputAmount"]) == len(args["inputToken"]), args
        zipped = set(zip(args["inputAmount"], args["inputToken"]))
        newa = []
        newt = []
        for z in zipped:
            if z[0] != "0":
                newa.append(z[0])
                newt.append(z[1])
        assert len(newa) == len(newt), (args, newa, newt)
        if len(newa) > 1:
            args["inputAmount"] = newa
            args["inputToken"] = newt
        elif len(newa) == 1:
            args["inputAmount"] = newa[0]
            args["inputToken"] = newt[0]
        else:
            raise Exception(f"{args['inputAmount']} {args['inputToken']}")
    if (
        "inputAmount" in args
        and "inputToken" in args
        and isinstance(args["inputAmount"], list)
        and isinstance(args["inputToken"], str)
    ):
        args["inputAmount"] = args["inputAmount"][0]
    if (
        "inputAmount" in args
        and "inputToken" in args
        and isinstance(args["inputAmount"], list)
        and isinstance(args["inputToken"], list)
    ):
        assert len(args["inputAmount"]) == len(args["inputToken"])
        newt = []
        newa = []
        for t, a in zip(args["inputToken"], args["inputAmount"]):
            if t == a and t == "all":
                continue
            else:
                newt.append(t)
                newa.append(a)
        assert len(newa) == len(newt)
        if len(newa) > 1:
            args["inputAmount"] = newa
            args["inputToken"] = newt
        elif len(newa) == 1:
            args["inputAmount"] = newa[0]
            args["inputToken"] = newt[0]
        else:
            raise Exception(f"{args['inputAmount']} {args['inputToken']}")


def args_amount_units2(args) -> None:
    """
    Clean up multi amount/inputAmountUnits/token pairs
    """
    if (
        "inputAmount" in args
        and "inputToken" in args
        and "inputAmountUnits" in args
        and isinstance(args["inputAmount"], list)
        and isinstance(args["inputToken"], list)
        and isinstance(args["inputAmountUnits"], list)
    ):
        assert len(args["inputAmount"]) == len(args["inputToken"]), (
            args["inputAmount"],
            args["inputToken"],
        )
        if len(args["inputAmount"]) != len(args["inputAmountUnits"]) or len(
            args["inputToken"]
        ) != len(args["inputAmountUnits"]):
            del args["inputAmountUnits"]
            return
        zipped = set(
            zip(args["inputAmount"], args["inputAmountUnits"], args["inputToken"])
        )
        newa = []
        newu = []
        newt = []
        for z in zipped:
            newa.append(z[0])
            newu.append(z[1])
            newt.append(z[2])
        assert len(newa) == len(newt) and len(newa) == len(newu), (newa, newu, newt)
        if len(newa) > 1:
            args["inputAmount"] = newa
            args["inputAmountUnits"] = newu
            args["inputToken"] = newt
        elif len(newa) == 1:
            args["inputAmount"] = newa[0]
            args["inputAmountUnits"] = newu[0]
            args["inputToken"] = newt[0]
        else:
            raise Exception(
                f"{args['inputAmount']} {args['inputAmountUnits']} {args['inputToken']}"
            )
    if (
        "inputAmountUnits" in args
        and "inputToken" in args
        and isinstance(args["inputAmountUnits"], list)
        and isinstance(args["inputToken"], str)
    ):
        args["inputAmountUnits"] = args["inputAmountUnits"][0]


def args_amount1(args, message: str, tokens):
    """
    Clean up amount
    """
    if "amount" in args and isinstance(args["amount"], str):
        if (
            args["amount"] != ""
            and args["amount"].lower() != "outputAmount".lower()
            and args["amount"] != "all"
            and args["amount"] != "half"
        ):
            if (
                args["amount"].lower() not in message.lower().split(" ")
                and "'" + args["amount"].lower() + "'" not in message.lower().split(" ")
                and re.sub(r"^0+", "", args["amount"].lower())
                not in message.lower().split(" ")
            ):
                del args["amount"]
            else:
                try:
                    _y = float(
                        args["amount"]
                        .replace("%", "")
                        .replace("$", "")
                        .replace(",", "")
                    )
                except Exception:
                    # print(e)
                    if args["amount"] in tokens and (
                        ("token" in args and args["token"] == "") or "token" not in args
                    ):
                        args["token"] = args["amount"]
                    del args["amount"]
    elif "amount" in args and isinstance(args["amount"], list):
        new = []
        for ia in args["amount"]:
            if (
                ia != ""
                and ia.lower() != "outputAmount".lower()
                and ia != "all"
                and ia != "half"
            ):
                if ia.lower() not in message.lower():
                    continue
                else:
                    try:
                        _y = float(
                            ia.replace("%", "").replace("$", "").replace(",", "")
                        )
                    except Exception:
                        # print(e)
                        continue
            new.append(ia)
        if len(new) > 1:
            args["amount"] = new
        elif len(new) == 1:
            args["amount"] = new[0]
        else:
            del args["amount"]


def args_amount0(args, message: str, tokens):
    """
    Clean up amount
    """
    if "inputAmount" in args and isinstance(args["inputAmount"], str):
        if (
            args["inputAmount"] != ""
            and args["inputAmount"].lower() != "outputAmount".lower()
            and args["inputAmount"] != "all"
            and args["inputAmount"] != "half"
        ):
            if (
                args["inputAmount"].lower() not in message.lower().split(" ")
                and "'" + args["inputAmount"].lower() + "'"
                not in message.lower().split(" ")
                and re.sub(r"^0+", "", args["inputAmount"].lower())
                not in message.lower().split(" ")
            ):
                del args["inputAmount"]
            else:
                try:
                    _y = float(
                        args["inputAmount"]
                        .replace("%", "")
                        .replace("$", "")
                        .replace(",", "")
                    )
                except Exception:
                    # print(e)
                    yy = args["inputAmount"].split(" ")
                    if len(yy) > 1:
                        try:
                            _y = float(
                                yy[0].replace("%", "").replace("$", "").replace(",", "")
                            )
                            args["inputAmount"] = yy[0]
                        except Exception:
                            # print(e)
                            del args["inputAmount"]
                    else:
                        if args["inputAmount"] in tokens and (
                            ("inputToken" in args and args["inputToken"] == "")
                            or "inputToken" not in args
                        ):
                            args["inputToken"] = args["inputAmount"]
                        del args["inputAmount"]
    elif "inputAmount" in args and isinstance(args["inputAmount"], list):
        new = []
        for ia in args["inputAmount"]:
            if (
                ia != ""
                and ia.lower() != "outputAmount".lower()
                and ia != "all"
                and ia != "half"
            ):
                if ia.lower() not in message.lower():
                    continue
                else:
                    try:
                        _y = float(
                            ia.replace("%", "").replace("$", "").replace(",", "")
                        )
                    except Exception:
                        yy = ia.split(" ")
                        if len(yy) > 1:
                            try:
                                _y = float(
                                    yy[0]
                                    .replace("%", "")
                                    .replace("$", "")
                                    .replace(",", "")
                                )
                                ia = yy[0]
                            except Exception:
                                # print(e)
                                continue
                        else:
                            continue
            new.append(ia)
        if len(new) > 1:
            args["inputAmount"] = new
        elif len(new) == 1:
            args["inputAmount"] = new[0]
        else:
            del args["inputAmount"]


def args_chain(args, message: str, updated, chains, shortened_chains):
    """
    Clean up chains to remove helper words
    """
    for argname in [
        "chainName",
        "sourceChainName",
        "destinationChainName",
        "targetChainName",
    ]:
        if argname not in args:
            continue
        if args[argname] == "":
            del args[argname]
            continue
        if len(updated["chains"]) == 0:
            if args[argname].lower() not in message.lower():
                if args[argname].lower() in shortened_chains:
                    args[argname] = shortened_chains[args[argname].lower()]
                elif args[argname].lower() in list(shortened_chains.values()):
                    continue
                else:
                    del args[argname]
                    continue
        else:
            if (
                args[argname].lower() not in message.lower()
                and args[argname].lower() not in updated["chains"]
            ):
                if args[argname].lower() in shortened_chains:
                    args[argname] = shortened_chains[args[argname].lower()]
                else:
                    del args[argname]
                    continue
        if args[argname].lower() not in chains:
            if args[argname].lower() in shortened_chains:
                args[argname] = shortened_chains[args[argname].lower()]
            elif args[argname].lower() != "all":
                del args[argname]
                continue


def args_protocols_pools(args, message: str, updated, protocols, name, tokens):
    """
    Clean up protocols and pools
    """

    if (
        "protocolName" in args
        and args["protocolName"] != ""
        and args["protocolName"] != "all"
        and args["protocolName"].lower() not in message.lower()
        and args["protocolName"].lower() not in updated["protocols"]
    ):
        if re.compile(r"0x[\da-fA-F]{40}|\w+\.eth\b").fullmatch(
            args["protocolName"].lower()
        ):
            args["poolName"] = args["protocolName"]
        del args["protocolName"]
    if "protocolName" in args and "position" in args["protocolName"].lower():
        args["protocolName"] = args["protocolName"].lower().replace("position", "")
        args["protocolName"] = args["protocolName"].lower().replace(" position", "")
    if (
        "protocolName" in args
        and args["protocolName"] != ""
        and args["protocolName"] != "all"
        and args["protocolName"].lower() not in protocols
    ):
        if name in ["claim", "lock"] and args["protocolName"].lower() in tokens:
            args["token"] = args["protocolName"].lower()
        if re.compile(r"0x[\da-fA-F]{40}|\w+\.eth\b").fullmatch(
            args["protocolName"].lower()
        ):
            args["poolName"] = args["protocolName"]
        del args["protocolName"]
    if (
        "poolName" in args
        and args["poolName"] != ""
        and args["poolName"] != "all"
        and args["poolName"].lower() not in message.lower()
        and args["poolName"].lower() not in updated["pools"]
    ):
        if "token" in args and args["token"] in args["poolName"]:
            pass
        else:
            del args["poolName"]
    if "poolName" in args and args["poolName"].replace(" ", "") in [
        "rewards",
        "position",
        "collateral",
    ]:
        del args["poolName"]
    # if "poolName" in args and args["poolName"] != "" and args["poolName"] != "all":
    # if (
    # name in ["claim", "lock"]
    # and args["poolName"].lower() in tokens
    # and "token" not in args
    # ):
    # args["token"] = args["poolName"].lower()
    # del args["poolName"]
    if (
        name in ["withdraw"]
        and "poolName" in args
        and args["poolName"].replace(" ", "") == "liquidity"
    ):
        if "token" not in args:
            args["token"] = args["poolName"].lower()
        del args["poolName"]


def args_token2(args, message: str, updated):
    """
    Clean up tokens to remove helper words
    """
    if "inputToken" in args and isinstance(args["inputToken"], str):
        if args["inputToken"] == "rewards":
            args["inputToken"] = "outputToken"
        if (
            args["inputToken"] != ""
            and args["inputToken"].lower() != "outputToken".lower()
            and args["inputToken"] != "all"
            and args["inputToken"].lower() not in message.lower()
            and args["inputToken"].lower() not in updated["tokens"]
        ):
            del args["inputToken"]
    elif "inputToken" in args and isinstance(args["inputToken"], list):
        new = []
        for it in args["inputToken"]:
            if it == "rewards":
                it = "outputToken"
            if (
                it != ""
                and it.lower() != "outputToken".lower()
                and it != "all"
                and it.lower() not in message.lower()
                and it.lower() not in updated["tokens"]
            ):
                # print(1)
                continue
            if "outputToken" in args and it == args["outputToken"]:
                # print(2)
                continue
            new.append(it)
        if len(new) > 1:
            args["inputToken"] = new
        elif len(new) == 1:
            args["inputToken"] = new[0]
        else:
            del args["inputToken"]
    if (
        "outputToken" in args
        and args["outputToken"].lower() not in message.lower()
        and args["outputToken"].lower() not in updated["tokens"]
        and args["outputToken"].lower() != "weth"
    ):
        del args["outputToken"]
    if "outputToken" in args and args["outputToken"].lower() == "gas":
        del args["outputToken"]
    if (
        "targetToken" in args
        and args["targetToken"].lower() not in message.lower()
        and args["targetToken"].lower() not in updated["tokens"]
    ):
        del args["targetToken"]


def args_tokens_protocols_chains(
    name, args, chains, message: str, updated, protocols, shortened_chains
):
    """
    Clean up tokens and protocols and chains
    """
    if (
        name != "bridge"
        and "protocolName" in args
        and "chainName" not in args
        and args["protocolName"].lower() in chains
    ):
        args["chainName"] = args["protocolName"]
        args["protocolName"] = "all"
    if (
        name == "bridge"
        and "protocolName" in args
        and "sourceChainName" not in args
        and args["protocolName"].lower() in chains
    ):
        args["sourceChainName"] = args["protocolName"]
        del args["protocolName"]
    if name != "bridge" and "protocolName" in args and "chainName" not in args:
        pn = args["protocolName"].lower().replace("token", "").replace(" ", "")
        if pn in chains or pn in updated["chains"] or pn in shortened_chains:
            args["chainName"] = (
                args["protocolName"].lower().replace("token", "").replace(" ", "")
            )
            del args["protocolName"]
    if (
        name == "bridge"
        and "protocolName" not in args
        and "sourceChainName" in args
        and args["sourceChainName"].lower() in protocols
    ):
        args["protocolName"] = args["sourceChainName"]
        del args["sourceChainName"]
    if "token" in args and isinstance(args["token"], str):
        if args["token"] == "rewards":
            args["token"] = "outputToken"
        if (
            args["token"] != ""
            and args["token"].lower() != "outputToken".lower()
            and args["token"] != "all"
            and args["token"].lower() not in message.lower()
            and args["token"].lower() not in updated["tokens"]
        ):
            del args["token"]
        if "token" in args and args["token"].lower() in [
            "positions",
            "position",
            "incentives",
            "pool",
            "vested",
        ]:
            del args["token"]
    if "inputToken" in args and isinstance(args["inputToken"], str):
        if args["inputToken"] == "rewards":
            args["inputToken"] = "outputToken"
        if args["inputToken"].lower() in [
            "positions",
            "position",
            "incentives",
            "pool",
            "vested",
        ]:
            del args["inputToken"]
    elif "token" in args and isinstance(args["token"], list):
        new = []
        for it in args["token"]:
            if it == "rewards":
                it = "outputToken"
            if (
                it != ""
                and it.lower() != "outputToken".lower()
                and it != "all"
                and it.lower() not in message.lower()
                and it.lower() not in updated["tokens"]
            ):
                continue
            if "outputToken" in args and it == args["outputToken"]:
                continue
            new.append(it)
        if len(new) > 1:
            args["token"] = new
        elif len(new) == 1:
            args["token"] = new[0]
        else:
            del args["token"]


def args_tokens_pools(args, updated, message) -> None:
    """
    Clean up tokens and pools to remove helper words
    """
    if "outputToken" in args and (
        "token" in args["outputToken"].lower()
        or "chain" in args["outputToken"].lower()
        or "protocol" in args["outputToken"].lower()
        or "coin" in args["outputToken"].lower()
    ):
        args["outputToken"] = args["outputToken"].lower().replace(" chain", "")
        args["outputToken"] = args["outputToken"].lower().replace("chain ", "")
        args["outputToken"] = args["outputToken"].lower().replace("_chain", "")
        args["outputToken"] = args["outputToken"].lower().replace(" token", "")
        args["outputToken"] = args["outputToken"].lower().replace("token ", "")
        args["outputToken"] = args["outputToken"].lower().replace("_token", "")
        args["outputToken"] = args["outputToken"].lower().replace("token", "")
        args["outputToken"] = args["outputToken"].lower().replace(" protocol", "")
        args["outputToken"] = args["outputToken"].lower().replace("_protocol", "")
        args["outputToken"] = args["outputToken"].lower().replace("protocol", "")
        args["outputToken"] = args["outputToken"].lower().replace(" coin", "")
        args["outputToken"] = args["outputToken"].lower().replace("_lp", " lp")
        if " lp" in args["outputToken"]:
            updated["tokens"].append(args["outputToken"])
        args["outputToken"] = re.sub("^ ", "", args["outputToken"].lower())
        args["outputToken"] = re.sub(" $", "", args["outputToken"].lower())
    if "targetToken" in args and (
        "token" in args["targetToken"].lower()
        or "protocol" in args["targetToken"].lower()
        or "coin" in args["targetToken"].lower()
    ):
        args["targetToken"] = args["targetToken"].lower().replace(" token", "")
        args["targetToken"] = args["targetToken"].lower().replace("token ", "")
        args["targetToken"] = args["targetToken"].lower().replace("_token", "")
        args["targetToken"] = args["targetToken"].lower().replace("token", "")
        args["targetToken"] = args["targetToken"].lower().replace(" protocol", "")
        args["targetToken"] = args["targetToken"].lower().replace("_protocol", "")
        args["targetToken"] = args["targetToken"].lower().replace("protocol", "")
        args["targetToken"] = args["targetToken"].lower().replace(" coin", "")
        args["targetToken"] = args["targetToken"].lower().replace("_lp", " lp")
        if " lp" in args["targetToken"]:
            updated["tokens"].append(args["targetToken"])
        args["targetToken"] = re.sub("^ ", "", args["targetToken"].lower())
        args["targetToken"] = re.sub(" $", "", args["targetToken"].lower())
    if "subject" in args and (
        "token" in args["subject"] or "protocol" in args["subject"].lower()
    ):
        args["subject"] = args["subject"].lower().replace(" token", "")
        args["subject"] = args["subject"].lower().replace("token ", "")
        args["subject"] = args["subject"].lower().replace("_token", "")
        args["subject"] = args["subject"].lower().replace("token", "")
        args["subject"] = args["subject"].lower().replace(" protocol", "")
        args["subject"] = args["subject"].lower().replace("_protocol", "")
        args["subject"] = args["subject"].lower().replace("protocol", "")
        args["subject"] = args["subject"].lower().replace(" coin", "")
        args["subject"] = args["subject"].lower().replace("_lp", " lp")
    if "value" in args and (
        "token" in args["value"] or "protocol" in args["value"].lower()
    ):
        args["value"] = args["value"].lower().replace(" token", "")
        args["value"] = args["value"].lower().replace("token ", "")
        args["value"] = args["value"].lower().replace("_token", "")
        args["value"] = args["value"].lower().replace("token", "")
        args["value"] = args["value"].lower().replace(" protocol", "")
        args["value"] = args["value"].lower().replace("_protocol", "")
        args["value"] = args["value"].lower().replace("protocol", "")
        args["value"] = args["value"].lower().replace(" coin", "")
        args["value"] = args["value"].lower().replace("_lp", " lp")
    if "chainName" in args and "chain" in args["chainName"].lower():
        args["chainName"] = args["chainName"].lower().replace(" chain", "")
    if "sourceChainName" in args and "chain" in args["sourceChainName"].lower():
        args["sourceChainName"] = args["sourceChainName"].lower().replace(" chain", "")
    if (
        "destinationChainName" in args
        and "chain" in args["destinationChainName"].lower()
    ):
        args["destinationChainName"] = (
            args["destinationChainName"].lower().replace(" chain", "")
        )
    if "targetChainName" in args and "chain" in args["targetChainName"].lower():
        args["targetChainName"] = args["targetChainName"].lower().replace(" chain", "")
    if "protocolName" in args and "protocol" in args["protocolName"].lower():
        args["protocolName"] = args["protocolName"].lower().replace(" protocol", "")
    if "poolName" in args:
        if (
            args["poolName"].lower() not in message.lower().split(" ")
            or args["poolName"].lower() == "pool"
        ):
            args["poolName"] = args["poolName"].lower().replace(" pool", "")
            args["poolName"] = args["poolName"].lower().replace("_pool", "")
            args["poolName"] = args["poolName"].lower().replace("pool", "")
        args["poolName"] = args["poolName"].lower().replace(" token", "")
        args["poolName"] = args["poolName"].lower().replace("token ", "")
        args["poolName"] = args["poolName"].lower().replace("_token", "")
        args["poolName"] = args["poolName"].lower().replace("token", "")
        args["poolName"] = args["poolName"].lower().replace(" protocol", "")
        args["poolName"] = args["poolName"].lower().replace("_protocol", "")
        args["poolName"] = args["poolName"].lower().replace("protocol", "")
        args["poolName"] = args["poolName"].lower().replace(" vault", "")
        args["poolName"] = args["poolName"].lower().replace("_vault", "")
        args["poolName"] = args["poolName"].lower().replace("vault", "")
        args["poolName"] = args["poolName"].replace("_lp", "")
        args["poolName"] = args["poolName"].replace(" lp", "")
        if args["poolName"] == "":
            del args["poolName"]


def args_token1(args, updated) -> None:
    """
    Clean up tokens to remove helper words
    """
    if "inputToken" in args:
        if isinstance(args["inputToken"], list):
            randomlist = True
            for c, z in enumerate(args["inputToken"]):
                if z[-1] != str(c + 1):
                    randomlist = False
            if randomlist:
                args["inputToken"] = "all"
        if (
            isinstance(args["inputToken"], str)
            and args["inputToken"].lower() == "tokens".lower()
        ):
            args["inputToken"] = "all"
        if (
            isinstance(args["inputToken"], str)
            and args["inputToken"].lower() != "outputToken".lower()
        ):
            args["inputToken"] = args["inputToken"].lower().replace(" chain", "")
            args["inputToken"] = args["inputToken"].lower().replace("chain ", "")
            args["inputToken"] = args["inputToken"].lower().replace("_chain", "")
            args["inputToken"] = args["inputToken"].lower().replace(" token", "")
            args["inputToken"] = args["inputToken"].lower().replace("token ", "")
            args["inputToken"] = args["inputToken"].lower().replace("_token", "")
            args["inputToken"] = args["inputToken"].lower().replace("token", "")
            args["inputToken"] = args["inputToken"].lower().replace(" protocol", "")
            args["inputToken"] = args["inputToken"].lower().replace("_protocol", "")
            args["inputToken"] = args["inputToken"].lower().replace("protocol", "")
            args["inputToken"] = args["inputToken"].lower().replace(" coin", "")
            args["inputToken"] = args["inputToken"].lower().replace("_lp", " lp")
            if " lp" in args["inputToken"]:
                updated["tokens"].append(args["inputToken"])
            args["inputToken"] = re.sub("^ ", "", args["inputToken"].lower())
            args["inputToken"] = re.sub(" $", "", args["inputToken"].lower())
        elif isinstance(args["inputToken"], list):
            new = []
            for tok in args["inputToken"]:
                if tok == "outputToken" or tok == "outputtoken":
                    new.append(tok)
                    continue
                if tok == "tokens":
                    new.append("all")
                    continue
                tok = tok.lower().replace(" chain", "")
                tok = tok.lower().replace("chain ", "")
                tok = tok.lower().replace("_chain", "")
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
                    updated["tokens"].append(tok)
                tok = re.sub("^ ", "", tok.lower())
                tok = re.sub(" $", "", tok.lower())
                new.append(tok)
            if len(new) > 1:
                args["inputToken"] = new
            elif len(new) == 1:
                args["inputToken"] = new[0]
            else:
                del args["inputToken"]


def args_token0(args, updated) -> None:
    """
    Clean up tokens to remove helper words
    """
    if "token" in args:
        if isinstance(args["token"], list):
            randomlist = True
            for c, z in enumerate(args["token"]):
                if z[-1] != c + 1:
                    randomlist = False
            if randomlist:
                args["token"] = "all"
        if isinstance(args["token"], str) and args["token"].lower() == "tokens".lower():
            args["token"] = "all"
        if (
            isinstance(args["token"], str)
            and args["token"].lower() != "outputToken".lower()
        ):
            args["token"] = args["token"].lower().replace(" chain", "")
            args["token"] = args["token"].lower().replace("chain ", "")
            args["token"] = args["token"].lower().replace("_chain", "")
            args["token"] = args["token"].lower().replace(" token", "")
            args["token"] = args["token"].lower().replace("token ", "")
            args["token"] = args["token"].lower().replace("_token", "")
            args["token"] = args["token"].lower().replace("token", "")
            args["token"] = args["token"].lower().replace(" protocol", "")
            args["token"] = args["token"].lower().replace("_protocol", "")
            args["token"] = args["token"].lower().replace("protocol", "")
            args["token"] = args["token"].lower().replace(" coin", "")
            args["token"] = args["token"].lower().replace("_lp", " lp")
            if " lp" in args["token"]:
                updated["tokens"].append(args["token"])
            args["token"] = re.sub("^ ", "", args["token"].lower())
            args["token"] = re.sub(" $", "", args["token"].lower())
        elif isinstance(args["token"], list):
            new = []
            for tok in args["token"]:
                if tok == "outputToken" or tok == "outputtoken":
                    new.append(tok)
                    continue
                if tok == "tokens":
                    new.append("all")
                    continue
                tok = tok.lower().replace(" chain", "")
                tok = tok.lower().replace("chain ", "")
                tok = tok.lower().replace("_chain", "")
                tok = tok.lower().replace("token", "")
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
                    updated["tokens"].append(tok)
                tok = re.sub("^ ", "", tok.lower())
                tok = re.sub(" $", "", tok.lower())
                new.append(tok)
            if len(new) > 1:
                args["token"] = new
            elif len(new) == 1:
                args["token"] = new[0]
            else:
                del args["token"]


def args_basic_process_early(name, args, message: str, tokens):
    """
    Delete or relabel basic hallucinations
    """
    if "inputAmountUnits" in args and "inputToken" in args:
        if args["inputAmountUnits"] == args["inputToken"]:
            del args["inputAmountUnits"]
    if "amount_units" in args and "token" in args:
        if args["amount_units"] == args["token"]:
            del args["amount_units"]
    if (
        "inputAmount" in args
        and isinstance(args["inputAmount"], list)
        and len(args["inputAmount"]) == 1
    ):
        args["inputAmount"] = args["inputAmount"][0]
    if (
        "inputAmountUnits" in args
        and isinstance(args["inputAmountUnits"], list)
        and len(args["inputAmountUnits"]) == 1
    ):
        args["inputAmountUnits"] = args["inputAmountUnits"][0]
    if (
        "inputToken" in args
        and isinstance(args["inputToken"], list)
        and len(args["inputToken"]) == 1
    ):
        args["inputToken"] = args["inputToken"][0]
    if name == "swap" and "amount" in args and "inputAmount" not in args:
        args["inputAmount"] = args["amount"]
        del args["amount"]
    if (
        name not in ["condition", "time"]
        and "operator" in args
        and (
            "value" not in args or "start_time" not in args or "recurrence" not in args
        )
    ):
        del args["operator"]
    if name == "bridge" and "chainName" in args and "sourceChainName" not in args:
        args["sourceChainName"] = args["chainName"]
        del args["chainName"]
    if (
        "amount_units" in args
        and "token" in args
        and isinstance(args["amount_units"], str)
        and isinstance(args["token"], str)
        and args["amount_units"].lower() == args["token"].lower()
    ):
        del args["amount_units"]
    if (
        "amount_units" in args
        and (
            "token" not in args
            or (
                "token" in args
                and isinstance(args["token"], str)
                and args["token"].replace("token", "") == ""
            )
        )
        and isinstance(args["amount_units"], str)
    ):
        args["token"] = args["amount_units"]
        del args["amount_units"]
    if "leverageMultiplier" in args and (
        args["leverageMultiplier"].lower() not in message.lower()
        or args["leverageMultiplier"].lower() == "all"
    ):
        del args["leverageMultiplier"]
    if (
        "amount" in args
        and isinstance(args["amount"], str)
        and args["amount"].lower() == "rewardsamount"
    ):
        args["amount"] = "outputAmount"
    if (
        "token" in args
        and isinstance(args["token"], str)
        and args["token"].lower() == "rewardstoken"
    ):
        args["token"] = "outputToken"
    if (
        "inputAmount" in args
        and isinstance(args["inputAmount"], str)
        and args["inputAmount"].lower() == "rewardsamount"
    ):
        args["amount"] = "outputAmount"
    if (
        "inputToken" in args
        and isinstance(args["inputToken"], str)
        and args["inputToken"].lower() == "rewardstoken"
    ):
        args["amount"] = "outputToken"
    if "recurrence" in args and args["recurrence"] and "type" in args["recurrence"]:
        args["recurrence"]["type"] = args["recurrence"]["type"].replace(
            "hourly", "hours"
        )
    if "recurrence" in args and args["recurrence"] and "type" in args["recurrence"]:
        args["recurrence"]["type"] = args["recurrence"]["type"].replace("daily", "days")
    if "recurrence" in args and args["recurrence"] and "type" in args["recurrence"]:
        args["recurrence"]["type"] = args["recurrence"]["type"].replace(
            "weekly", "weeks"
        )
    if "recurrence" in args and args["recurrence"] and "type" in args["recurrence"]:
        args["recurrence"]["type"] = args["recurrence"]["type"].replace(
            "monthly", "months"
        )
    if "amount_units" in args and args[
        "amount_units"
    ].lower() not in message.lower().split(" "):
        del args["amount_units"]
    if (
        "amount_units" in args
        and args["amount_units"].lower().replace("_", "").replace(" ", "")
        == "token".lower()
    ):
        del args["amount_units"]
    if "inputAmountUnits" in args:
        if isinstance(args["inputAmountUnits"], str):
            delete = False
            if args["inputAmountUnits"].lower() not in message.lower().split(" "):
                delete = True
            if (
                args["inputAmountUnits"].lower().replace("_", "").replace(" ", "")
                == "token".lower()
            ):
                delete = True
            if (
                "outputToken" in args
                and args["inputAmountUnits"].lower() == args["outputToken"].lower()
            ):
                delete = True
            if (
                args["inputAmountUnits"] not in tokens
                and args["inputAmountUnits"] != "usd"
            ):
                delete = True
            if delete:
                del args["inputAmountUnits"]
        elif isinstance(args["inputAmountUnits"], list):
            new = []
            for au in args["inputAmountUnits"]:
                if au.lower() not in message.lower().split(" "):
                    continue
                if au.lower().replace("_", "").replace(" ", "") == "token".lower():
                    continue
                if "outputToken" in args and au.lower() == args["outputToken"].lower():
                    continue
                if au not in tokens and au != "usd":
                    continue
                new.append(au)
            if new == []:
                del args["inputAmountUnits"]
            elif len(new) == 1:
                args["inputAmountUnits"] = new[0]
            else:
                args["inputAmountUnits"] = new
    if (
        "amount" in args
        and (args["amount"] == "0.5" or args["amount"] == ["0.5"])
        and (
            "half" in message.lower().split(" ")
            or "'half'" in message.lower().split(" ")
        )
    ):
        args["amount"] = "half"
    if (
        "inputAmount" in args
        and (args["inputAmount"] == "0.5" or args["inputAmount"] == ["0.5"])
        and (
            "half" in message.lower().split(" ")
            or "'half'" in message.lower().split(" ")
        )
    ):
        args["inputAmount"] = "half"
    # if "outputToken" in args and "protocolName" in args and args["outputToken"].lower() == args["protocolName"].lower():
    # del args["protocolName"]
    if (
        "period" in args
        and "period" not in message.lower()
        and args["period"].lower() not in message.lower()
        and args["period"] != "forever"
    ):
        del args["period"]
    if (
        "outputToken" in args
        and args["outputToken"] == "token"
        and "inputAmountUnits" in args
    ):
        args["outputToken"] = args["inputAmountUnits"]
        del args["inputAmountUnits"]
    if (
        "inputAmountUnits" in args
        and (
            "inputToken" not in args
            or (
                "inputToken" in args
                and isinstance(args["inputToken"], str)
                and args["inputToken"].replace("token", "") == ""
            )
        )
        and isinstance(args["inputAmountUnits"], str)
    ):
        args["inputToken"] = args["inputAmountUnits"]
        del args["inputAmountUnits"]
    if (
        "inputAmountUnits" in args
        and (
            "inputToken" not in args
            or (
                "inputToken" in args
                and isinstance(args["inputToken"], list)
                and len(args["inputToken"]) > 0
                and args["inputToken"][0].replace("token", "") == ""
            )
        )
        and isinstance(args["inputAmountUnits"], list)
    ):
        args["inputToken"] = args["inputAmountUnits"]
        del args["inputAmountUnits"]
    if (
        "inputAmountUnits" in args
        and "inputToken" in args
        and isinstance(args["inputToken"], str)
        and isinstance(args["inputAmountUnits"], str)
        and (
            args["inputAmountUnits"].lower() == args["inputToken"].lower()
            or args["inputToken"].lower() == "outputtoken"
        )
    ):
        del args["inputAmountUnits"]
    if (
        "inputAmountUnits" in args
        and "inputAmount" in args
        and isinstance(args["inputAmount"], str)
        and isinstance(args["inputAmountUnits"], str)
        and (
            args["inputAmountUnits"].lower() == args["inputAmount"].lower()
            or args["inputAmount"].lower() == "outputamount"
        )
    ):
        del args["inputAmountUnits"]
    # if "end_time" in args and "recurrence" in args:
    # if str(args["recurrence"]["interval"]) in args["end_time"].split(" "):
    # del args["end_time"]
    # if not any(
    # element in message.lower() for element in ["every", "each", "recurrence"]
    # ):
    # del args["recurrence"]
    if "start_time" in args and args["start_time"] == "now":
        del args["start_time"]
    if "outputToken" in args and args["outputToken"] == "usd":
        del args["outputToken"]


async def preprocessing(
    message: str,
    rxtk: str,
    rxtk2: str,
    rxact: str,
    rxch: str,
    keywords,
    tokens,
    protocols,
    chains,
    shortened_chains,
    pools,
    orderedtokens,
    actions,
    confusing1,
    confusing2,
    confusing3,
    support_chat=False,
):
    """
    Perform preprocessing of message to help AI understanding.
    Remove extra words, label entities, add helper words.
    """
    om = message
    message = message.lower()
    message = re.sub(r"^\s+|\s+$", "", message)
    message = re.sub(r"\.\s*" + rxact, r". then \g<act>", message)
    message = re.sub(rxch + r"\s* l2", r"\g<chn>", message)
    message = first_preprocess(message, shortened_chains)
    rxnm = r"(?P<num>\d+\.\d*|\d+|\.\d+)"
    rxnm2 = r"(?P<num2>\d+\.\d*|\d+|\.\d+)"
    rxdr = r"(?P<dur>min(ute)?s?|s(ec)?(ond)?s?|h(ou)?rs?|days?|w(ee)?ks?)"
    rxar = r"(?P<addr>0x[\da-fA-F]{40}|\w+\.eth\b)"
    message = first_regex(message, rxnm, rxnm2)
    message = pre_entities(message)
    # print(0, message)
    message = compress_number_in_sentence(message)
    rxwds = r"(?P<wd>[a-zA-Z]+)"
    second_regex_pattern(rxnm, rxtk, rxtk2, rxdr, rxar, rxwds)
    # print('sr start', time.time())
    assert "eth" in tokens - protocols - keywords
    # print(1, message)
    message = fast_second_regex(message, rxnm, rxtk, rxtk2, rxdr, rxar, rxwds)  # slow
    if "usd" in message:
        message = fast_second_regex(
            message, rxnm, rxtk, rxtk2, rxdr, rxar, rxwds
        )  # slow
    # print(2, message)
    # sys.exit(0)
    # print('sr end', time.time())
    message, updated, error_messages = await post_entities(
        rxar,
        rxtk,
        rxtk2,
        message,
        keywords,
        tokens,
        protocols,
        chains,
        pools,
        orderedtokens,
        actions,
    )
    # print('lpp start', time.time())
    message = last_preprocess(
        om,
        message,
        tokens,
        confusing1,
        confusing2,
        confusing3,
        protocols,
        chains,
        pools,
        support_chat=support_chat,
    )  # slow
    # print('lpp end', time.time())
    message = message.replace("  ", " ")
    message = message.replace("  ", " ")
    message = message.replace(" ,", ",")
    message = message.replace('"', "")
    message = message.replace("\\", "")
    message = message.replace("usd usd", "usd")
    message = re.sub(r"^\s+|\s+$", "", message)
    message = re.sub(r"^\s+|\s+$", "", message)
    print("original message: ", om)
    print("preprocessed message: ", message)
    return (
        om,
        message,
        updated,
        error_messages,
    )


def compress_number_in_sentence(sentence):
    """
    Compresses a number within a sentence by removing commas, spaces, and converting to integer.

    Args:
        sentence: The sentence containing the number (e.g., "buy 19,000,000 eth").

    Returns:
        The sentence with the compressed number (e.g., "buy 19000000 eth") or original sentence if nothing to compress.
    """
    pattern = r"\b(?<!\.)\d+(?:[,\s]\d+)*\b"  # Match digits with optional comma or space-separated digits
    match = re.search(pattern, sentence)
    if match:
        number_str = match.group()  # Extract the matched number string
        compressed_number = int(
            re.sub(r"[,\s]", "", number_str)
        )  # Remove commas and spaces
        return sentence.replace(number_str, str(compressed_number))
    else:
        return sentence


@lru_cache
def second_regex_pattern(rxnm, rxtk, rxtk2, rxdr, rxar, rxwds):
    return re.compile(
        r"(?P<g1>"
        + r"\$"
        + rxwds.replace("wd", "g1tkn")
        + r"\s+"
        + r")|"
        + r"(?P<g2>"
        + r"\$"
        + rxwds.replace("wd", "g2tkn")
        + r"\.$"
        + r")|"
        + r"(?P<g3>"
        + r"\$"
        + rxwds.replace("wd", "g3tkn")
        + r"$"
        + r")|"
        + r"(?P<g4>"
        + r"s\*"
        + rxtk.replace("tkn", "g4tkn")
        + r"\b"
        + r")|"
        + r"(?P<g5>"
        + "\$"
        + r")|"
        + r"(?P<g6>\b"
        + rxnm.replace("num", "g6num")
        + r"e\b"
        + r")|"
        + r"(?P<g7>"
        + " "
        + rxnm.replace("num", "g7num")
        + rxtk.replace("tkn", "g7tkn")
        + " "
        + r")|"
        + r"(?P<g8>"
        + rxtk.replace("tkn", "g8tkn")
        + r"\s*-\s*"
        + rxtk2.replace("tkn2", "g8tkn2")
        + " "
        + r")|"
        + r"(?P<g88>"
        + rxtk.replace("tkn", "g88tkn")
        + r"\s*of\s*"
        + rxtk2.replace("tkn2", "g88tkn2")
        + r"\b"
        + r")|"
        + r"(?P<g89>"
        + rxtk.replace("tkn", "g89tkn")
        + r"\s*of\s*"
        + rxar.replace("addr", "g89ar")
        + r")|"
        + r"(?P<g9>"
        + rxnm.replace("num", "g9num")
        + rxdr.replace("dur", "g9dur")
        + r")|"
        + r"(?P<ga>\b"
        + rxnm.replace("num", "ganum")
        + r"\s*(k|thousand)\b"
        + r")|"
        + r"(?P<gb>\b"
        + rxnm.replace("num", "gbnum")
        + r"\s*(m|mn|million)\b"
        + r")|"
        + r"(?P<gc>\b"
        + rxnm.replace("num", "gcnum")
        + r"\s*(b|bn|billion)\b"
        + r")|"
        + r"(?P<gd>\b"
        + rxnm.replace("num", "gdnum")
        + r"\s*(d)\b"
        + r")|"
        + r"(?P<ge>"
        + r"\s+(lp)\s+(?P<gelp>\d*\.\d+|\d+)\b"
        + r")|"
        + r"(?P<gf>"
        + r"^lp\s+(?P<gflp>\d*\.\d+|\d+)\b"
        + r")|"
        + r"(?P<gg>"
        + r"\s+(lp)\s+"
        + rxtk.replace("tkn", "ggtkn")
        + r"\b"
        + r")|"
        + r"(?P<gh>"
        + r"^lp\s+"
        + rxtk.replace("tkn", "ghtkn")
        + r"\b"
        + r")|"
        + r"(?P<gi>"
        + r"(?P<gipt>,|\.)\s*(lp)\s+into\b"
        + r")|"
        + r"(?P<gj>"
        + r"(?P<gjpt>\s+|-)\s*"
        + rxtk.replace("tkn", "gjtkn")
        + r"\s*lp\b"
        + r")|"
        + r"(?P<gk>"
        + r"(?<!amount_units )"
        + r"\busd\s*(worth)?\s*(of)?\s*"
        + rxtk.replace("tkn", "gktkn")
        + r"\b"
        + r")|"
        + r"(?P<gl>"
        + r"(?<!amount_units )\b"
        + rxtk.replace("tkn", "gltkn")
        + r"\s*worth\s*of\s*"
        + rxtk2.replace("tkn2", "gltkn2")
        + r"\b"
        + r")|"
        + r"(?P<gm>"
        + r"\s+(?P<gmcp>\d*\.\d+|\d+)\s*market\s*cap\b"
        + r")|"
        + r"(?P<gn>"
        + r"(?<!amount_units )"
        + r"\busd\s*(worth)?\s*(of)?\s*"
        + rxar.replace("addr", "gkar")
        + r"\b"
        + r")|"
        + r"(?P<go>"
        + r"\b"
        + rxtk.replace("tkn", "gotkn")
        + r"\s+"
        + rxtk2.replace("tkn2", "gotkn2")
        + r"\s+pool"
        + r"\b"
        + r")"
    )


def second_regex(message, rxnm, rxtk, rxtk2, rxdr, rxar):
    """
    Label entities to help AI understanding
    """
    message = re.sub(r"\$" + rxtk + r"\s+", r"\g<tkn> token ", message)
    message = re.sub(r"\$" + rxtk + r".$", r"\g<tkn> token.", message)
    message = re.sub(r"\$" + rxtk + r"$", r"\g<tkn> token", message)
    message = message.replace("$", "")
    message = re.sub(r"\b" + rxnm + r"e\b", r"\g<num> eth", message)
    message = re.sub(" " + rxnm + rxtk + " ", r" \g<num> \g<tkn> ", message)
    message = re.sub(
        rxtk + r"\s*-\s*" + rxtk2 + " ", r"\g<tkn>-\g<tkn2> pool ", message
    )
    message = re.sub(
        rxtk + r"\s*of\s*" + rxar, r"\g<tkn> of output token \g<addr>", message
    )
    message = re.sub(r"\b" + rxnm + rxdr, r"\g<num> \g<dur>", message)
    message = re.sub(r"\b" + rxnm + r"\s*k\b", r"\g<num>000", message)
    message = re.sub(r"\b" + rxnm + r"\s*(m|mn)\b", r"\g<num>000000", message)
    message = re.sub(r"\b" + rxnm + r"\s*(b|bn)\b", r"\g<num>000000000", message)
    message = re.sub(r"\b" + rxnm + r"\s*(d)\b", r"\g<num> day", message)
    message = re.sub(r"\s+(lp)\s+(\d*\.\d+|\d+)\b", r" deposit \2", message)
    message = re.sub(r"^lp\s+(\d*\.\d+|\d+)\b", r"deposit \1", message)
    message = re.sub(r"\s+(lp)\s+" + rxtk + r"\b", r" deposit \g<tkn>", message)
    message = re.sub(r"^lp\s+" + rxtk + r"\b", r"deposit \g<tkn>", message)
    message = re.sub(
        r"(,|\.)\s*(lp)\s+into\b", r"\1 deposit outputtokens into", message
    )
    message = re.sub(r"(\s+|-)\s*" + rxtk + r"\s*lp\b", r"\1\g<tkn>_lp token", message)
    message = re.sub(
        r"\s*usd\s+(worth of\s+)?" + rxtk + r"\b",
        r" amount_units usd of \g<tkn>",
        message,
    )
    message = re.sub(r"\s+(\d*\.\d+|\d+)\s*market\s*cap\b", r" \1_market_cap", message)
    return message


def fast_second_regex(message, rxnm, rxtk, rxtk2, rxdr, rxar, rxwds):
    """
    Label entities to help AI understanding
    """
    return re.sub(
        second_regex_pattern(rxnm, rxtk, rxtk2, rxdr, rxar, rxwds),
        lambda m: (
            m.group("g1tkn") + " token "
            if m.group("g1")
            else "" + m.group("g2tkn") + " token."
            if m.group("g2")
            else "" + m.group("g3tkn") + " token"
            if m.group("g3")
            else "" + "'s*" + m.group("g4tkn") + "' token"
            if m.group("g4")
            else "" + ""
            if m.group("g5")
            else "" + m.group("g6num") + " eth"
            if m.group("g6")
            else "" + " " + m.group("g7num") + " " + m.group("g7tkn") + " "
            if m.group("g7")
            else "" + m.group("g8tkn") + "-" + m.group("g8tkn2") + " pool "
            if m.group("g8")
            else "" + m.group("g88tkn") + " of output token " + m.group("g88tkn2")
            if m.group("g88")
            else "" + m.group("g89tkn") + " of output token " + m.group("g89ar")
            if m.group("g89")
            else "" + m.group("g9num") + " " + m.group("g9dur")
            if m.group("g9")
            else "" + str(float(m.group("ganum")) * 1000)
            if m.group("ga")
            else "" + str(float(m.group("gbnum")) * 1000000)
            if m.group("gb")
            else "" + str(float(m.group("gcnum")) * 1000000000)
            if m.group("gc")
            else "" + m.group("gdnum") + " day"
            if m.group("gd")
            else "" + " deposit " + m.group("gelp")
            if m.group("ge")
            else "" + "deposit " + m.group("gflp")
            if m.group("gf")
            else "" + " deposit " + m.group("ggtkn")
            if m.group("gg")
            else "" + "deposit " + m.group("ghtkn")
            if m.group("gh")
            else "" + m.group("gipt") + " deposit outputtokens into"
            if m.group("gi")
            else "" + m.group("gjpt") + m.group("gjtkn") + "_lp token"
            if m.group("gj")
            else "" + " amount_units usd of " + m.group("gktkn")
            if m.group("gk")
            else ""
            + " amount_units "
            + m.group("gltkn")
            + " worth of "
            + m.group("gltkn2")
            if m.group("gl")
            else "" + " " + m.group("gmcp") + "_market_cap"
            if m.group("gm")
            else "" + " amount_units usd of " + m.group("gkar")
            if m.group("gn")
            else "" + m.group("gotkn") + "-" + m.group("gotkn2") + " pool"
            if m.group("go")
            else ""
        ),
        message,
    )


async def entities(fast=False, production=False):
    """
    Get entities to use throughout code
    """
    actions = [
        "transfer",
        "swap",
        "bridge",
        "deposit",
        "withdraw",
        "claim",
        "borrow",
        "lend",
        "repay",
        "stake",
        "unstake",
        "long",
        "short",
        "lock",
        "unlock",
        "vote",
        "buy",
        "sell",
        "close",
    ]
    shortened_chains = get_shortened_chains()
    protocols, tokens, chains, keywords, pools, orderedtokens = await get_entities(
        actions, fast, production=production
    )
    confusing1 = protocols & tokens
    confusing2 = chains & tokens
    confusing3 = set(actions) & tokens
    # rxnm = r'(?P<num>\d+)'
    return (
        actions,
        protocols,
        tokens,
        chains,
        shortened_chains,
        keywords,
        pools,
        orderedtokens,
        confusing1,
        confusing2,
        confusing3,
    )


def first_regex(message: str, rxnm, rxnm2):
    """
    Add labels to numbers
    """
    message = re.sub(r"\s+0" + "," + rxnm2, r" 0.\g<num2>", message)
    message = re.sub(rxnm + "," + rxnm2, r"\g<num>\g<num2>", message)
    message = re.sub(rxnm + "," + rxnm2, r"\g<num>\g<num2>", message)
    message = re.sub(rxnm + "," + rxnm2, r"\g<num>\g<num2>", message)
    # message = re.sub(r'\$(\d*\.\d+|\d+)', r'price \1', message)
    message = re.sub(r"(is|at)\s*\$(\d*\.\d+|\d+)", r"== \2 usd", message)
    message = re.sub(r"\$(\d*\.\d+|\d+)([kKmMbBtT]?)", r"\1\2 usd", message)
    message = re.sub(r"(\d*\.\d+|\d+)([kKmMbBtT]?)\$", r"\1\2 usd", message)
    message = re.sub(r"(\d*\.\d+|\d+)\s*\%", r"'\1%'", message)
    message = re.sub(r"(\d*\.\d+|\d+)x\s+", r"'\1x' ", message)
    message = re.sub(rxnm + ":" + rxnm2, r"time \g<num>:\g<num2>", message)
    message = re.sub(r"usdc\.e(?!pool|/|-)", r"'usdc.e' token", message)
    message = message.replace("usdc.e", "'usdc.e'").replace("''", "'")
    return message


def pre_entities(message: str):
    """
    Basic clean up
    """
    message = message.replace("exactly", "")
    message = message.replace("''", "'")
    message = message.replace("  ", " ")
    message = message.replace("  ", " ")
    message = message.replace("eth mainnet", "ethereum")
    message = message.replace("ethereum mainnet", "ethereum")
    message = message.replace("mainnet", "ethereum")
    message = message.replace("zk sync", "zksync")
    message = message.replace("unwind borrow", "repay")
    message = message.replace("reduce borrow", "repay")
    message = message.replace("exit borrow", "repay")
    message = message.replace("repay borrow", "repay")
    message = message.replace("borrow position", "position")
    return message


def last_preprocess(
    om: str,
    message: str,
    tokens,
    confusing1,
    confusing2,
    confusing3,
    protocols,
    chains,
    pools,
    support_chat=False,
):
    """
    Label entities and clean up
    """
    snapshot = message
    # dont need to label pools because will always say ____ pool in a prompt
    # for o in pools:
    # message = re.sub(r"protocol\s*" + f"{re.escape(o)} token", f"{o} pool", message)
    # message = message.replace(f" {o} ", f" {o} pool ")
    # message = message.replace(f" {o}. ", f" {o} pool.")
    # message = message.replace(f" {o},", f" {o} pool,")
    # message = re.sub(f" {re.escape(o)}$", f" {o} pool", message)
    # message = re.sub(f" {re.escape(o)}\.$", f" {o} pool.", message)
    # message = re.sub(f" {re.escape(o)}\?$", f" {o} pool?", message)
    # message = re.sub(f"^{re.escape(o)} ", f"{o} pool ", message)
    rxnmb = r"(?P<nmb>(\d+\.\d*|\d+|\.\d+|all|half)\s*)"
    for p in protocols:
        if p not in message:
            continue
        message = re.sub(
            rxnmb + f" {p} (?!positions)", r"\g<nmb>" + f" {p} token ", message
        )
        message = re.sub(f"for {p}$", f"for {p} token", message)
        message = message.replace(f" {p} ", f" {p} protocol ")
        message = message.replace(f" {p}. ", f" {p} protocol. ")
        message = message.replace(f" {p},", f" {p} protocol,")
        message = re.sub(f" {p}$", f" {p} protocol", message)
        message = re.sub(f" {p}\.$", f" {p} protocol.", message)
        message = re.sub(f" {p}\?$", f" {p} protocol?", message)
        message = re.sub(f"^{p} ", f"{p} protocol ", message)
    message = message.replace("protocol token", "token")
    for t in tokens:
        if (
            t in confusing1
            or t in confusing2
            or t in confusing3
            or t not in message
            or t == "position"
        ):
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
    for c in chains:
        if c not in message or c == "bnb" or c == "matic":
            continue
        message = message.replace(f" {c} ", f" {c} chain ")
        message = message.replace(f" {c}. ", f" {c} chain. ")
        message = message.replace(f" {c}.", f" {c} chain.")
        message = message.replace(f" {c},", f" {c} chain,")
        message = re.sub(f" {c}$", f" {c} chain", message)
        message = re.sub(f" {c}\.$", f" {c} chain.", message)
        message = re.sub(f" {c}\?$", f" {c} chain?", message)
        message = re.sub(f"^{c} ", f"{c} chain ", message)
    message = message.replace("token pool", "pool")
    message = message.replace("protocol pool", "protocol")
    message = message.replace("pool pool", "pool")
    message = message.replace("pool token", "token")
    message = message.replace("pool lp", "lp")
    message = message.replace("protocol finance", "protocol")
    if support_chat:
        message = message.replace("protocol token", "protocol")
    else:
        message = message.replace("protocol token", "")
    message = message.replace("protocol cartel", "cartel")
    message = message.replace("protocol vault", "vault")
    message = message.replace("protocol dao", "dao protocol")
    message = message.replace("protocol exchange", "protocol")
    message = message.replace("protocol protocol", "protocol")
    message = message.replace("token finance", "finance")
    message = message.replace("token staking", "staking")
    message = message.replace("token cartel", "cartel")
    if support_chat:
        message = message.replace("token protocol", "protocol")
    else:
        message = message.replace("token protocol", "")
    if support_chat:
        message = message.replace("token chain", "chain")
        message = message.replace("chain token", "token")
    else:
        message = message.replace("token chain", "")
        message = message.replace("chain token ", " ")
    message = message.replace("token rewards token", "rewards token")
    message = message.replace("token token", "token")
    message = message.replace("chain network", "chain")
    message = message.replace("chain chain", "chain")
    olist = [x for x in om.lower().split(" ") if x != ""]
    mlist = [x for x in message.lower().split(" ") if x != ""]
    newmlist = []
    # startidx = 0
    for ix, ml in enumerate(mlist):
        newmlist.append(ml)
        occs = [i + 1 for i, x in enumerate(olist) if x == ml]
        for occ in occs:
            if occ == len(olist):  # or occ <= startidx:
                continue
            next = olist[occ]
            if next in ["token", "chain", "protocol", "pool"]:
                if ix < len(mlist) - 1 and mlist[ix + 1] not in [
                    "token",
                    "chain",
                    "protocol",
                    "pool",
                ]:
                    newmlist.append(next)
                    # startidx = occ
                    # break
    message = " ".join(newmlist)
    olist = [x for x in snapshot.lower().split(" ") if x != ""]
    mlist = [x for x in message.lower().split(" ") if x != ""]
    newmlist = []
    # startidx = 0
    for ix, ml in enumerate(mlist):
        newmlist.append(ml)
        occs = [i + 1 for i, x in enumerate(olist) if x == ml]
        for occ in occs:
            if occ == len(olist):  # or occ <= startidx:
                continue
            next = olist[occ]
            if next in ["token", "chain", "protocol", "pool"] and ml not in [
                "token",
                "chain",
                "protocol",
                "pool",
            ]:
                if ix < len(mlist) - 1 and mlist[ix + 1] not in [
                    "token",
                    "chain",
                    "protocol",
                    "pool",
                ]:
                    newmlist.append(next)
                    # startidx = occ
                    # break
    message = " ".join(newmlist)
    message = message.replace("token token", "token")
    message = message.replace("token token", "token")
    message = message.replace("token rewards token", "rewards token")
    message = message.replace(" _price", "_price")
    message = message.replace("protocol rewards", "rewards")
    message = message.replace("pool pool", "pool")
    message = message.replace("pool lp", "pool")
    message = message.replace("chain chain", "chain")
    message = message.replace("arbitrum chain one chain", "arbitrum chain")
    message = message.replace("arbitrum chain one", "arbitrum chain")
    message = message.replace("zksync chain era chain", "zksync chain")
    message = message.replace("zksync chain era", "zksync chain")
    message = message.replace("bsc chain smart chain", "bsc chain")
    message = message.replace("bsc chain smart", "bsc chain")
    return message


def first_preprocess(message: str, shortened_chains):
    nl = []
    ml = message.split(" ")
    for word in ml:
        if len(word) == 0:
            # print(message, ml)
            continue
        if word[-1] == "$":
            nl.append("$" + word[:-1])
        else:
            nl.append(word)
    message = " ".join(nl)
    message = message.replace("'", "")
    message = message.replace(" the ", " ")
    message = message.replace(" that ", " ")
    message = message.replace("compound my ", "claim and redeposit ")
    message = message.replace("compound them ", "claim and redeposit them ")
    message = message.replace(" my ", " ")
    message = message.replace(" me ", " ")
    for sc in shortened_chains:
        message = message.replace(f"on {sc} ", f"on {shortened_chains[sc]} ")
        message = re.sub(rf"on {sc}$", f"on {shortened_chains[sc]}", message)
    rxwd = r"(?P<wkdy>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    message = re.sub(
        r"\s+(\d+)\s+(january|february|march|april|may|june|july|august|september|october|november|december)",
        r" time \2 \1",
        message,
    )
    message = re.sub(
        r"(each|every)\s*" + rxwd, r"time \g<wkdy> recurrence weekly", message
    )
    message = re.sub(r"\s*on\s*" + rxwd, r" on start_time \g<wkdy>", message)
    message = re.sub(
        r"(each|every|once a)\s*" + r"week", r"time recurrence weekly", message
    )
    message = re.sub(
        r"(each|every|once a)\s*" + r"day", r"time recurrence daily", message
    )
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
    message = message.replace(", whenever ", ", condition ")
    message = message.replace(",whenever ", ", condition ")
    message = message.replace(" whenever ", ", condition ")
    message = re.sub("^whenever ", "condition ", message)
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
    message = message.replace(" convert ", " swap ")
    message = message.replace(",convert ", ",swap ")
    message = message.replace(".convert ", ".swap ")
    message = re.sub("^convert ", "swap ", message)
    message = re.sub(
        r"\bwrap\s+(all\s+)?(half\s+)?(of\s+)?(my\s+)?(the\s+)?eth",
        "swap eth to weth" if "half" not in message else "swap half eth to weth",
        message,
    )
    message = re.sub(
        r"\bwrap\s+(?P<num>\d+\.\d*|\d+|\.\d+)\s+eth",
        r"swap \g<num> eth to weth",
        message,
    )
    message = re.sub(
        r"\bunwrap\s+(all\s+)?(half\s+)?(of\s+)?(my\s+)?(the\s+)?weth",
        "swap weth to eth" if "half" not in message else "swap half weth to eth",
        message,
    )
    message = re.sub(
        r"\bunwrap\s+(?P<num>\d+\.\d*|\d+|\.\d+)\s+weth",
        r"swap \g<num> weth to eth",
        message,
    )
    message = message.replace(" wrap ", " swap ")
    message = message.replace(".wrap ", ".swap ")
    message = message.replace(",wrap ", ",swap ")
    message = re.sub("^wrap ", "swap ", message)
    message = message.replace(" unwrap ", " swap ")
    message = message.replace(".unwrap ", ".swap ")
    message = message.replace(",unwrap ", ",swap ")
    message = re.sub("^unwrap ", "swap ", message)
    # message = message.replace(" sell", " swap")
    # message = re.sub("^sell ", "swap ", message)
    message = message.replace(" dump", " swap")
    message = re.sub("^dump ", "swap ", message)
    message = message.replace(" grab", " claim")
    message = re.sub("^grab ", "claim ", message)
    message = message.replace(" harvesting", " claiming")
    message = re.sub("^harvesting ", "claiming ", message)
    message = message.replace(" harvest", " claim")
    message = re.sub("^harvest ", "claim ", message)
    message = message.replace(" move ", "transfer ")
    message = re.sub("^move ", "transfer ", message)
    message = message.replace("relock", "lock")
    message = message.replace("revest", "lock")
    message = message.replace("supply", "lend")
    message = message.replace("unvest", "unlock")
    message = message.replace("restake", "stake")
    message = message.replace("remove", "withdraw")
    message = message.replace(" bid ", " buy ")
    message = message.replace(" send ", " transfer ")
    message = re.sub("^send ", "transfer ", message)
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
    message = message.replace("payback", "repay")
    message = message.replace("price impact", "slippage")
    message = message.replace("arbitrum one", "arbitrum")
    message = message.replace("binance", "bsc")
    message = message.replace("binancesmartchain", "bsc")
    message = message.replace("linea-mainnet", "linea")
    message = message.replace("homestead", "ethereum")
    message = message.replace("zk sync", "zksync")
    message = message.replace("zksync era", "zksync")
    message = message.replace("ether.fi", "etherfi")
    message = message.replace("dollars", "usd")
    message = message.replace("equivalent of", "worth of")
    unders = [
        " below ",
        " is under ",
        " lt ",
        " lte ",
        " less than ",
        " is less than ",
        " less_than ",
        " lessThan ",
        " lessthan ",
        " goes below ",
        " sub ",
        " is at or below ",
    ]
    overs = [
        " above ",
        " is over ",
        " gt ",
        " gte ",
        " greater than ",
        " is greater than ",
        " greater_than ",
        " greaterThan ",
        " greaterthan ",
        " goes above ",
        " is at or above ",
        " exceeds ",
        " surpasses ",
    ]
    equals = [
        " equal ",
        " hits ",
        " hit ",
        " is at ",
        " reaches ",
        " near ",
        " goes to ",
    ]
    notequals = [" notequals "]
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
    return message


async def post_entities(
    rxar,
    rxtk,
    rxtk2,
    message: str,
    keywords,
    tokens,
    protocols,
    chains,
    pools,
    orderedtokens,
    actions,
):
    # print(f"Original message: {message}")

    # Identify words using rxm in the filtered message
    premsglist = re.findall(
        r"\b(?![0-9]+x\b)[\w-]+\b",
        re.sub(f"({rxar}|pm$|am$)", r"", message)
        .replace(",", "")
        .replace(".", "")
        .replace("%", "")
        .replace(" pm ", " ")
        .replace(" am ", " ")
        .replace("pm ", " ")
        .replace("am ", " "),
    )
    # print(f"Words identified in the message: {premsglist}")

    # Identify unrecognized words not in any provided lists or digits
    all_known_words = set().union(
        keywords, tokens, protocols, chains, pools, actions, orderedtokens
    )
    ordinal_pattern = re.compile(r"\d+(st|nd|rd|th)")
    version_pattern = re.compile(r"v\d+")
    pre_unrecognized_words = [
        word
        for word in premsglist
        if word not in all_known_words
        and not word.isdigit()
        and not ordinal_pattern.match(word)
        and not version_pattern.match(word)
    ]
    unrecognized_words = []
    for puw in pre_unrecognized_words:
        rpuw = puw.replace("_lp", "").replace("_pool", "")
        if len(re.sub(rxtk + "-" + rxtk2, "", rpuw)) < 2:
            continue
        else:
            if "_" in puw or "-" in puw:
                continue
            unrecognized_words.append(puw)
    if unrecognized_words:
        print(f"Unrecognized words: {unrecognized_words}")

    # Make a megalist of all known words for comparison excluding tokens
    levenshtein_search_list = set().union(protocols, chains, pools, actions, keywords)

    # Apply Levenshtein distance
    threshold = 2  # Adjustable threshold for similarity
    replacements = {}
    error_messages = []
    for word in unrecognized_words:
        min_distance = float("inf")
        potential_replacements = []
        for known_word in levenshtein_search_list:
            distance = levenshtein_distance(word, known_word)
            if distance <= threshold:
                if distance < min_distance:
                    potential_replacements = [known_word]
                    min_distance = distance
                elif distance == min_distance:
                    potential_replacements.append(known_word)
        if len(potential_replacements) == 0:
            error_messages.append(
                f"2cf524f4486f40a2a427b46b08db5595{word} is not supported! Refer to the sidenav for supported actions, protocols, and chains, or the autosuggestions for best practices when prompting."
            )
        if len(potential_replacements) == 1:
            replacements[word] = potential_replacements[0]
        elif len(potential_replacements) > 1:
            for pr in potential_replacements:
                if pr in protocols or pr in chains or pr in actions:
                    replacements[word] = pr
                    break
            else:
                print(potential_replacements)
                error_messages.append(
                    f"2cf524f4486f40a2a427b46b08db5595{word} is not recognized, you might have a typo. If this was a token, try using the contract address."
                )

    if replacements:
        print(f"Replacements: {replacements}")

    # Apply direct replacements
    for word, replacement in replacements.items():
        message = message.replace(f" {word} ", f" {replacement} ")
        message = message.replace(f" {word}.", f" {replacement}.")
        message = message.replace(f" {word},", f" {replacement},")
        message = message.replace(f" {word}?", f" {replacement}?")
        message = re.sub(f" {word}$", f" {replacement}", message)
        message = re.sub(f"^{word} ", f"{replacement} ", message)

    final_replacements = replacements.copy()

    # print(f"Replacements: {replacements}")
    # print(f"Final replacements: {final_replacements}")

    # Prepare the updated entities dictionary
    updated = {
        "tokens": [word for word in final_replacements.values() if word in tokens],
        "protocols": [
            word for word in final_replacements.values() if word in protocols
        ],
        "chains": [word for word in final_replacements.values() if word in chains],
        "pools": [word for word in final_replacements.values() if word in pools],
        "actions": [word for word in final_replacements.values() if word in actions],
        "orderedtokens": [
            word for word in final_replacements.values() if word in orderedtokens
        ],
    }
    # print(f"Updated: {updated}")

    # Log error messages if any
    if error_messages:
        for error_message in error_messages:
            print(error_message)

    return message, updated, error_messages


async def get_entities(
    actions: list[str], fast: bool = False, production: bool = False
):
    keywords = get_keywords(actions)
    protocols = get_protocols(production)
    chains = get_chains(production)
    pools = get_pools(production)
    tokens0 = get_tokens(keywords)
    ve = await make_request_with_retries(entities_url)
    if not ve:
        with open("verified-entities.json", "r", encoding="utf-8", errors="igonre") as f:
            ve = json.load(f)
    for a in ve["actions"]:
        keywords.add(a.lower())
    for p in ve["protocols"]:
        if "name" in p:
            if not p["name"].isdigit():
                protocols.add(p["name"].lower())
        if "pools" in p:
            for k, v in list(p["pools"].items()):
                if isinstance(v, dict):
                    for kv in list(v.keys()):
                        if not kv.isdigit():
                            pools.add(kv.lower())
    for c in ve["chains"]:
        if "name" in c:
            if not c["name"].isdigit():
                chains.add(c["name"].lower())
        if "tokens" in c:
            for t in c["tokens"]:
                if (
                    t["symbol"].lower() not in keywords
                    and t["symbol"] != ""
                    and not any(
                        z in t["symbol"].lower()
                        for z in [
                            ".",
                            "+",
                            "*",
                            "?",
                            "^",
                            "$",
                            "(",
                            ")",
                            "[",
                            "]",
                            "{",
                            "}",
                            "|",
                            "\\",
                        ]
                    )
                    and not all(zz.isdigit() for zz in t["symbol"])
                ):
                    tokens0.append(t["symbol"].lower())
                if (
                    t["name"].lower() not in keywords
                    and t["name"] != ""
                    and not any(
                        z in t["name"].lower()
                        for z in [
                            ".",
                            "+",
                            "*",
                            "?",
                            "^",
                            "$",
                            "(",
                            ")",
                            "[",
                            "]",
                            "{",
                            "}",
                            "|",
                            "\\",
                        ]
                    )
                    and not all(zz.isdigit() for zz in t["name"])
                ):
                    tokens0.append(t["name"].lower())
    orderedtokens = tokens0
    tokens = set(tokens0)
    assert "thruster" in protocols
    assert "debridge" in protocols
    return protocols, tokens, chains, keywords, pools, orderedtokens


@app.get("/status")
async def status():
    return JSONResponse(content={"status": "OK"})


@app.post("/process-message")
async def process_message(message: dict):
    try:
        input_message = message["message"]
    except KeyError:
        raise HTTPException(
            status_code=400, detail="Field 'message' is required in the request body."
        )
    try:
        user_address = message["user_address"]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="Field 'user_address' is required in the request body.",
        )

    results = await process_message_ai(
        input_message,
        user_address=user_address,
        production=True,
        default_chain=(
            message["connected_chain"] if "connected_chain" in message else ""
        ),
    )
    print(results)

    return JSONResponse(content={"status": "OK", "data": results})


async def fetch_chunks(url) -> Any:
    async with httpx.AsyncClient(
        timeout=timeout,
        proxies={
            "http": "http://web.hm.sivalik.com:1443",
            "https": "http://web.hm.sivalik.com:1443",
        },
    ) as client:
        response = await client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
            },
        )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    text = " ".join(soup.stripped_strings)
    return pd.DataFrame(
        {"url": url, "chunk": re.findall(r"(?:.|\n){1,987}(?:\. |$)", text)}
    )


async def process_urls(urls):
    return pd.concat([await fetch_chunks(url) for url in urls], ignore_index=True)


async def get_embeddings(
    list_of_text: list[str], model="text-embedding-ada-002"
) -> list[list[float]]:
    return [
        d.embedding
        for d in (
            await client.embeddings.create(
                input=[text.replace("\n", " ") for text in list_of_text], model=model
            )
        ).data
    ]


def distances_from_embeddings(
    query_embedding: list[float], embeddings: list[list[float]]
) -> list[list]:
    return [
        spatial.distance.cosine(query_embedding, embedding) for embedding in embeddings
    ]


def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    return np.argsort(distances)


async def embedding_search(query, df):
    embedding = (await get_embeddings([query]))[0]
    df["embedding"] = await get_embeddings(df["chunk"])
    df["similarity"] = distances_from_embeddings(embedding, df["embedding"])
    idx = indices_of_nearest_neighbors_from_distances(df["similarity"]).tolist()
    return df.iloc[idx]


async def google_search(query) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(
            "https://www.google.com/search",
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
            },
            params={"q": query},
        )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    return [
        url
        for url in [link.get("href") for link in soup.find_all("a")]
        if re.search(r"^https://", url or "")
        if not re.search("google", url or "")
    ]


async def research(query: str):
    urls0 = await google_search(query)
    urls = [url for url in urls0 if re.search(r"beincrypto", url) is None]
    df = await process_urls(urls[:2])
    res = await embedding_search(query, df)
    prompt = res["chunk"].head(3).tolist()
    messages: list[openai.types.chat.ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "give a concise answer to the question based on the context",
        },
        {"role": "user", "content": f"question: {query}\n\ncontext: {prompt}"},
    ]
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        temperature=1,
        timeout=10,
    )
    print(response)
    return response.choices[0].message.content


async def suite(step=0, prompt=1, save=0):
    """
    examples that our model should achieve 100% accuracy on
    """
    if step == 0 or step == 1:
        # "Hey can you get some eth, convert it, deposit in kwenta, open a short trade with this leverage"
        # "Ok i want to 3x leverage GLP and zap out of USDC into some other asset thats support on Dolomite"
        # "carry out x no of swap on the dapp daily for 1 month when gas is less than 30"
        # "Claim LP staking rewards/airdrops to add back to the LP"
        # "Bridge from mainnet and long a coin in one sweep."
        # "buying on camelot"
        # "buy x amount usdc , x amount usdt, x amount DAI, and stake in trycrypto curve pool"
        # "Vote on my THENA position every week on Wednesday"
        # "Deposit 100 ARB into Plutus and stake LP for PLS, then lock PLS"
        # "withdraw position from trader joe"
        # "voting solidly forks every week and claiming rewards of the same next day"
        # "Vote on all my positions once a week"
        # "vote on the most optimal pair on solidly every wednesday at this time"
        # "process RLBTRFLY rewards bi weekly....then take the weth i receive and deposit into blur vault."
        # "Grab rewards from Balancer and convert to ETH every week"
        # "spot a farm on defi lama and realize you dont have funds on lets say zkSync"
        # "rebalancing pools and farms on different chains"
        # "There are many different markets to claim rewards and reinvest as well for LP positions"
        usermsgs = [
            "Swap 1 ETH for USDC",
            "Swap 1 ETH for USDC on Uniswap",
            "Bridge 1 USDT from Base to Arbitrum",
            "Bridge 1 USDT from Base to Arbitrum on Hop Protocol",
            "Transfer 10 DAI to niyant.eth",
            "Swap 1 ETH for USDC on Ethereum then bridge to Arbitrum",
            "Bridge 1 ETH from Ethereum to Optimism then buy USDC",
            "Bridge 1 WETH from Base to Ethereum and deposit in Aave",
            "Swap 10 ETH for USDC when gas is below 20",
            "Swap 10 ETH for USDC when ETH is below 1600",
            "Swap 10 ETH for USDC in twelve hours",
            "Swap 10 ETH for USDC at 5pm",
            "Swap 10 ETH for USDC in twelve hours, repeating every twelve hours",
            "Swap 10 ETH for USDC at 5pm, repeating every 1 hour",
            "Deposit all my WETH into Aave",
            "Swap all my WETH into USDC",
            "Buy USDT with all my WETH",
            "Bridge all my WETH to Base",
            "Withdraw 0.1 ETH from Compound and buy OP",
            "Bridge 3 ETH to Avalanche and buy OHM",
            "Use 3 ETH to buy OHM on Avalanche",
            "Buy GRAIL with 4 WETH",
            "Bridge all my tokens on Canto to Ethereum",
            "Open a short trade on Kwenta on BTC with 3 ETH with 3x leverage",
            "Withdraw from all my positions, convert to WETH, and bridge to Arbitrum",
            "Swap eth for usdt, swap usdc for usdt, bridge usdt to arbitrum",
            "When gas is below 10, deposit 100 USDC into Morpho",
            "At 10am tomorrow, transfer 200 USDC to 0x2B605C2a76EE3F08a48b4b4a9d7D4dAD3Ed46bf3",
            "Stake 10 ETH on Rocket Pool",
            "Harvest all my positions on Arbitrum",
            "Swap all my tokens on Optimism to WETH and bridge to Arbitrum",
            "Swap 1 ETH to USDC, bridge to Arbitrum, deposit into JonesDAO, then deposit LP into Rodeo",
            "Bridge 1 ETH to Base, swap half to USDC, deposit into Kyber eth-usdc pool",
            "Harvest my MMF yield farms and automatically stake MMF every day at 8am",
            "Harvest my positions every Wednesday",
            "3x leverage long GLP with 1000 USDC on GMX and swap 1000 USDC into UNI",
            "Swap 500 DAI for WBTC every day for a month when gas is less than 30",
            "Claim and restake rewards from all my positions every Monday",
            "Bridge 200 USDT from Ethereum to Base and buy PEPE",
            "harvesting on camelot",
            "Using 2 ETH buy USDC, USDT, and DAI, then deposit into Curve tricrypto pool",
            "Vote on my THENA position every week on Wednesday",
            "Deposit 100 ARB into Plutus, stake LP for PLS, then lock PLS",
            "withdraw position from trader joe",
            "Vote on Solidly every Wednesday and claim Solidly rewards every Thursday",
            "Borrow 1000 USDC from Compound and deposit into Aave",
            "Borrow 1000 USDC from Compound and deposit into Aave",
            "Withdraw from all my positions on Ethereum and convert everything to ETH",
            "Vote, harvest, and restake all my positions every day",
            "Vote on all my positions every Sunday",
            "vote on the most optimal pair on solidly every wednesday at this time",
            "Harvest and restake all my positions every week",
            "Process rewards on Redacted Cartel, swap to WETH, and deposit into Blur, biweekly",
            "grab weekly rewards from ve(3,3) DEXes and convert them to ETH",
            "Grab rewards from Balancer and convert to ETH every week",
            "Bridge 1000 USDC from Ethereum to zkSync and deposit into PancakeSwap",
            "Withdraw 100 USDC from JonesDAO, bridge to Ethereum, and deposit it into Yearn",
            "Claim and redeposit rewards on all my protocols every week on Wednesday",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 2:
        # "Want to DCA every week"
        # "straddling buy / sell at specific prices"
        # "on Bitcoin, Buy at x price, Sell at y, Rinse and repeat for 48 hours"
        # "Compound my Stargate position"
        # "await process_message_ai("for my position in pendle, if it reaches $1.50, sell it. Buy back at $1.20", "", prompt, save)"
        # "I want to stake my arb, please give me 3 options"
        # "You get staking rewards as stables and wETH-GRAIL LP.; You gotta exit the LP then sell each individually for whatever you want"
        # "conditional triggers (e.g. depeg on a stable)"
        # "take eth and buy Jones then pair into lp position on sushi then take the lp token and trade it for plsjones"
        # "faster arbitrage across chains"
        # "arbitrage process and he had to bridge + swap + send it everywhere + go to velodrome"
        # "claiming rewards and compounding g into the pool"
        # "setting up DCA buying based on time and buy/sells on price levels"
        # "withdrawing from LPs/staking"
        # "harvesting rewards, but seeking them and twapping into new tokens I want to accumulate"
        # "PT-GLP and money markets for PT-GLP; Something were lacking is looping strategies on PT; Would love to set up a prompt and have users execute operation in one go"
        # "Will short the short maturity and long the long maturity; Hes arbing the yield between the two pools"
        # "Wants to loop with pendle to earn yield; have to go pendle to buy PT, then go to a market to borrow PT, then go back to pendle to buy PT"
        # "claim rewards from camelot, 3 transactions to claim, plus two additional transactions to convert dust grail into xgrail then allocate to dividend"
        # "relocked RL BTRFLY, CLAIM rewards also"
        # "Withdraw usdc from aave if compound usdc interest rate > aave."
        # "If bitcoin hits 15k, buy eth"
        # "marketbuy eth if bitcoin touches 15k"
        # "autocompounding any position"
        # "setting up take profit/stop loss or optimizing pools"
        # "Swing trading non-pooled pairs based on their ratio (dpx/rdpx)"
        # "Unstaking and selling when ratio between a liquid derivative and the native asset hits certain ratio, being able to reverse that operation (say plsdpx on Plutus)"
        # "Bridge from Arbitrum to Base and buy COIN when gas is under 12"
        # "consolidate entire portfolio into ETH and get it onto arb when gas is low"
        usermsgs = [
            "Buy BTC with 1 ETH every week",
            "Buy BTC with 1 ETH when BTC is at or below $25000 and sell 0.2 BTC for ETH when BTC is at or above $30000, forever",
            "Claim and restake my Chronos position every week on Monday",
            "Bridge 4 USDT to Base",
            "Swap 3 ETH to USDC and deposit into the ETH-USDC pool on Dolomite",
            "Open a 2x ETH long on GMX with 1000 USDC",
            "Vote on my Thena position every Wednesday",
            "Withdraw 2 ETH from my ETH-USDC pool position on Camelot",
            "Claim STG from my Stargate positions, swap to WETH, and deposit back into Stargate",
            "For my pendle token, if it reaches $1.50, sell it for USDC. If it reaches $1.20, buy back with USDC",
            "Stake my ARB on Arbitrum",
            "Harvest my Balancer position and stake the rewards",
            "Withdraw half the liquidity from my Dolomite USDC-USDT position",
            "Claim wETH-GRAIL LP rewards from Camelot and sell for USDC",
            "Sell all my USDC for ETH if USDC goes below $0.95",
            "Buy JONES with half my ETH, deposit into the ETH-JONES pool on Sushi, then trade LP for plsJones",
            "Buy ETH with 1000 USDC on Uniswap on Ethereum, bridge to Optimism, and sell for USDC on Velodrome",
            "Swap 5000 USDC for ETH on Sushiswap on Ethereum, bridge to Base, sell ETH for USDC on KyberSwap, bridge USDC back to mainnet",
            "Buy WBTC with ETH on Uniswap and sell it for ETH on Sushiswap",
            "Buy WBTC with 1 ETH on Uniswap",
            "Swap 1 ETH for USDT",
            "Swap XYZ for ABC on Pancakeswap in 35 minutes",
            "Swap XYZ for ABC on Pancakeswap at 11 PM UST",
            "Claim my Camelot rewards, swap to USDC, and deposit back into Camelot",
            "Buy WBTC with 1 ETH every Sunday",
            "Withdraw from my Lodestar position",
            "Harvest all my rewards on Arbitrum and buy ETH",
            "Lend 5 ETH, borrow 100 PT, then deposit 100 PT and 100 GLP into the PT-GLP pool on Pendle",
            "Lend 250 SMP and borrow 125 LMP on Pendle",
            "Claim rewards from Camelot, swap rewards and GRAIL into xGRAIL, then deposit xGRAIL into Camelot",
            "Claim Redacted rewards and relock BTRFLY",
            "Withdraw all my USDC from Aave and deposit into Compound",
            "If bitcoin goes below 15k, buy ETH",
            "Claim Stargate rewards, swap to ETH, redeposit",
            "Buy ETH with 5000 USDC. Sell ETH for USDC if the price goes below 1000 or above 3000",
            "Buy DPX with RDPX if the price of DPX/RDPX <= 0.8",
            "Unstake all my plsDPX and sell it for DPX if the price of plsDPX/DPX < 0.95",
            "Bridge 4 ETH from Arbitrum to Base and buy COIN when gas is under 12",
            "Swap all my tokens to ETH and buy ARB when gas is below 10",
            "Swap all my tokens to ETH and transfer to niyant.eth on mainnet",  # turn everything into eth and send to preset addy on main
            "Swap half of all my tokens to ETH and transfer to niyant.eth on mainnet",
            "Can you use my DAI to purchase sWeed",
            "Use DAI to purchase sWeed",
            "Deposit 50 USDC and 50 USDT into DODO Finance USDC-USDT pool, then every Friday claim DODO and swap to USDT",  # You LP USDC-USDT and earn DODO at 7-8% APY which you can dump for stables
            "When my ETH balance hits 1, buy 0.5 ETH worth of SAINT once the price of SAINT/ETH is under 20 and gas under 15",
            "Stake STG on Stargate, then every Friday claim and restake rewards",  # Stake STG on stargate, every Friday claim and restake rewards every week on Friday
            "Swap 10 ETH for USDC when the ETH market cap is below 20",
            "Swap 10 ETH for USDC when the market cap of ETH is below 1600",
            "Swap 10 ETH for USDC when my ETH balance is below 1600",
            "When my Camelot rewards balance is greater than 10 ETH, swap to USDC",
            "Deposit 10 ETH into the Yearn yETH pool when the APY is 15%",
            "Deposit 10 ETH into the yETH pool on Yearn when APY is 15%",
            "buy 1 eth",
            "swap all my wbtc for usdt at 12pm tomorrow or if usdt price goes below $0.9",
            "swap xyz for abc when gas is below 14 and abc market cap is below 100 eth",
            "swap all my eth for usdc when gas is less than 20 or eth/usdt goes above 2000",
            "swap all my wbtc for usdt when my eth balance is greater than 2 or eth/dai goes above 2000",
            "Swap 1 eth for usdc with 2% slippage",
            "Swap 1 eth for usdc with max 3% slippage",
            "Bridge 1 eth for usdc with 2% slippage",
            "Bridge 1 eth for usdc then swap to dai with max 2% slippage",
            "swap all my usdt for dai",
            "swap all my usdt and usdc for dai",
            "Withdraw all my USDC and USDT from Rodeo, convert to ETH, and bridge all of it to mainnet",  # Withdraw from Rodeo, convert to ETH, and bridge all of it to mainnet
            "deposit 10 usdc and usdt into the uniswap usdc-usdt pool",
            "swap all my dai and half my usdt for usdc on curve",
            "Swap 10 ETH for USDC when ETH is below 1600",
            "Buy ETH with 1000 USDC when ETH/USDC is less than 2000",
            "Buy ETH with 1000 USDC when ETH/USDC price is less than 2000",
            "Buy ETH with 1000 USDC when the price of ETH/USDC is less than 2000",
            "when my USDC balance hits 3000, send it to arkham.eth",
            "bridge 1 eth from etheruem to arbitum",
            "deposit 1 gril into camlot",
            "buy ustd with 2 ETH",
            "when my dolomite rewards hit $2000, swap them for usdc",
            "when my dolomite rewards balance hits $2000, swap them for usdc",
            "Once my Plutus rewards hit 2 ETH, claim rewards and transfer to person.eth",
            "sell 2 eth",
            "buy 0.1 eth",
            "buy 10 usdc",
            "Buy btc with eth when it is at 20000",
            "when its at 20000, buy btc",
            "once btc hits 20000, sell all my btc at midnight",
            "at 2am, swap all my wbtc for eth if gas is less than 15",
            "swap 1 eth to usdc with 1.5% slippage when gas is less than 10",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 3:
        usermsgs = [
            "take my 1000 usdc, convert it into eth, deposit it into my prologue nft on spice, then borrow 60%",
            "bridge 5 eth from arbitrum to ethereum mainnet and long $pepe",
            "swap everything i own on eth mainnet to $eth and bridge it all to arbitrum",
            "withdraw 1 eth from my jonesdao position and ape $jesus",
            "swap by 2 $eth for $geth, convert to 1 $reth and 1 $steth, stake both on rocketpool",
            "deposit 0.33 ETH and 500 USDT in the ETH/USDT LP on Uniswap",
            "short $uni on october 16 at 12pm est",
            "buy $eth with 7500 $usdc when $eth is $1400 and sell it all at $1600",
            "Lend WETH as collateral on polylend, borrow WETH on polylend",  # "Lend WETH as collateral on polylend, borrow WETH my WETH collateral with a max LTV of 80% and a borrow APY of -12.24%, farm with WETH on polylend",
            "Lend ETH as collateral on wing-finance which earns a Supply APY of 90.79%. Borrow USDT against your ETH collateral with a max LTV of 85% and a borrow APY of -1.55% (The interest you need to pay). Farm with USDT on paraspace-lending-v1 which earns 26.65%.",
            "swap all of my usdc to eth and eth to usdc on woofi on the zksync network",  # "swap all of my usdc to eth and eth to usdc 20 times over on woofi on the zksync network",
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
            "withdraw all funds from the spice finance prologue vault, swap to eth, bridge all funds to arbitrum when gas is <10 swap everything except gas into steth",  # "withdraw all funds from the Spice Finance Prologue Vault into ETH, bridge all funds to Arbitrum when gas is <10 swap everything except gas into stETH",
            "send 0.05 eth to 0x6955e7216e8d9d2ab2ca5ca5e31ccf7307e9d59f when gas is < 10",
            "send all my funds to 0x6955e7216e8d9d2ab2ca5ca5e31ccf7307e9d59f",
            "swap my ohm to steth, bridge everything to arbitrum",  # "get out of my OHM and into stETH, bridge everything to arbitrum.",
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
            "repay my borrow position on dolomite when borrow apy rises above 69%",  # "Unwind my borrow position on Dolomite when borrow APY rises above 69%",
            "Pull my liquidity on Uniswap if price falls below X",
            "Sell half of asset X as soon as price hits $2",
            "When gas is below 8 bridge 0.1 eth to Zksync",
            'At exactly 10pm tomorrow buy "Random CA" with 40 gwei',
            'Swap 0.05 eth to usdt and send the swapped usdt to "wallet address"',
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
            "whenever the eth price reaches $1,500, buy eth",  # "Help me purchase ETH whenever the price reaches $1,500, use defillama's meta-dex aggregator and give me the best rate",
            "vote for the most profitable strategy on any >10m mcap -pool in the thena ve(3,3) voting pools. Do this at 11:55. At 12:05, collect the rewards of the previous voting epoch and exchange them for DOGE at market prices, using swap aggregator swap.defillama.com with 0.5% slippage.",
            "Buy 1 ETH when ETH price is $1550",
            "LP 2 ETH into balancer and compound the rewards every 3 days",
            "At exactly 19:05 UTC, bridge $50 ETH to Starknet ETH",
            "if ETH is over 1800, sell for USDC",
            "bridge [amount] ETH from Ethereum to Arbitrum using the most cost-effective method. then, convert it to WETH.",
            "Bridge [amount] WETH from Arbitrum One back to Ethereum and then trade it for USDC.",
            "Exchange all existing tokens in my wallet for ETH. Once, finished send it to [CEX deposit wallet]",
            "when gas is below 10, harvest the eth yield from Dolomite and deposit the eth to Rodeo",  # "When gas is below 10, harvest the Eth yield from xyz and deposit the Eth to abc",
            "Set stop loss for ETH on arbitrum chain. Sell 0.1 ETH when price goes lower than 1500",
            "Set limit orders to buy ETH each time the price dips 10% from the current price, buy for 100 USDC each time.",
            "each monday, claim my vested token from the Dolomite protocol, and sell it for ETH at market price.",  # "Each monday, claim my vested token from the XYZ protocol / or give smart contract address, and sell it for ETH at market price.",
            "each monday, claim my vested tokens, and stake them in Jones jUSDC pool",  # "Each monday, claim my vested tokens, and stake them in xyz pool",
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
            "Send/transfer 1 eth on base and give me usdc on zksync",
            "bridge 1 eth from base to zksync and swap to usdc",
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
            "buy dai every 37 min for 2 days",  # "buy X token every 37 min for 2 days",
            "deposit usdc and eth into camelot usdc/eth pool, stake the spnft into a nitro pool",  # "open a position for usdc/eth on camelot, lock the spnft and deposit into a nitro pool",
            "sell everything from multiple chains into jusdc",
            "when Eth price hits $1500 buy 3 eth worth using my $usdc Balance",
            "swap my $bitcoin for usdc",
            "disperse 0.1E to these 10 wallets",
            "When gas is below 7, swap 0.01 ETH to USDC, repay 50% of USDC loan on Aave, withdraw 50% of supplied ETH on Aave.",
            "Buy 1 eth with USDT if price goes to 1200$",
            "Swap 100 of token A for token B",
            "swap for z token",
            "swap weth into eth",  # "Unwrap Weth into Eth, find the fastest & cheapest dex or dex aggregator to do it in",
            "trade 1 baby bear for eth then swap that eth for usdc and send that usdc over to the arbitrum network with hop exchange",
            "Disperse ETH into few wallets",
            "every saturday at 18:00 claim incentives from my velodrome lock positions. sell all incentives into velo. lock that velo into the same lock position.",  # "Every saturday at 18:00 claim incentives from my velodrome lock positions, if the fees are more than $100 worth claim them as well. Sell all incentives and fees claimed if any into VELO. Lock that VELO into the same lock position.",
            "If token X goes -30%, sell to USDC",
            "buy xxx$ of $TICKER with stablecoins/from my usdc",
            "bridge 20$ from polygon to arbitrum",  # "bridge x$ from polygon to arbitrum",
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
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 4:
        usermsgs = [
            "on GMX v2, when ETH price is near x, can you deposit x USDC collateral to avoid liquidation",
            "can you gather all the ETH that I have spread around different chains and bridge it in the most cost-efficient form to Arbitrum",
            "Bridge eth to zkynsc swap half of eth to usdc then stake it",
            "If Ldo goes down 10% or has a onchain liquidation bid it",
            "Claim liveTHE rewards once they are worth greater than 10$ and compound them back into liveTHE",
            "Claims LiveTHE rewards once they are greater than 10$ and swap to USDT",
            "Vote for the highest yielding bribe on thena on Wednesday at 2355 UTC",
            "bridge 0.1 ETH from Base to Linea",
            "Swap $50 of ETH into USDC on Kyberswap",
            "Swap asset x for y, at % max price impact",
            "Unvest 300000 Trove from Nitro Cartel and swap to USDC; Revest remaining Trove at 16:00 UTC",
            "Deposit USDC into Jones jUSDC and deposit to Rodeo Finance USDC: (contract address)",
            "bridge eth from mainnet to arb when gas is below 15",
            "claim rewards from Thena/Velodrome/Aerodrome, convert them to ETH and bridge them to Ethereum",
            "Whenever I have over $4000 in Ethereum, send $3500 to the gemini address, but only do this once a week at most",
            "Claim and sell and STG rewards",
            "collect weekly synthetix rewards, claim as soon as gas is below 10",
            "Bridge 500 usdc  each to linea, zk sync and base when gas is below 10",
            "harvest my position on sushi and also my position on camelot",
            "harvest on sushi and sell/restake.",
            "Stake Btrfly. Restake rewards on may 15th (etc)",
            "carry out x no of swap on the dapp  daily for 1 month when gas is less than 30",
            "Swap 100 of 0x... to y ",
            "Bridge all my eth from Arbitrum to ethereum",
            "Swap 200 usdt for btc when btc price is 26700$",
            "Sell x% of x$coin when price is x",
            "short Eth if it goes below or touches xxx price",
            "Deposit $500 usdc and 0.35eth into an lp on uniswap v3",
            "deposit 0.35 eth into aave",
            "borrow $400 usdc and swap to $BITCOIN",
            "when gas is below 15, deposit 10 eth as collateral in Aave. Then take out a loan of 10,000 usdc against that eth. Then take that 10,000 usdc and deposit it into Rollbit",
            "deposit 2 eth as collateral into Rage trade on arbitrum. Once Eth price goes below $1,750, long Eth with 5x leverage",
            "buy 5,000 usdc worth of eth and then bridge it to zk sync era using orbiter bridge. Then at noon each day for the next week swap $500 of eth for usdc and then swap back $500 worth of usdc for eth",
            "Whenever $WINR (0xD77B108d4f6cefaa0Cae9506A934e825BEccA46E) falls below 3$, swap 2eth for it",
            "swap my $bitcoin on the dex with the least amount of slippage",
            "bridge eth to arb when gas is sub .5$ and swap back when arb hits .90",
            "Unwind my borrow position on Dolomite when borrow APY rises above 69 %",
            "Sell X% of Y coin (or insert contract address?) every (day of week) at 12am utc on (insert preferred DEX).",
            "swap USDC for Arb if the price reaches $0.90 (ideally it would be smart enough to recognize that I mean once it DROPS to $0.90)",
            "If ETH hits 1200 USD open a 12x long on GMX",
            "send 0.01E to [wallet/contract address] every hour for 5 days",
            "Bridge 3.2 ethereums to mainnet",
            "Send $1000 to [insert wallet] every",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 5:
        usermsgs = [
            "when MKR hits $1500, buy 1 MKR with ETH",
            "when MKR trades at $1500, buy 1 MKR with ETH",
            "give me an alert if $BNB break the 200$ support",
            "give me an alert whenever stablecoin inflow see a significant growth over 30D period (atleast +10%)",
            "buy xyz if its market cap falls below 50mn over the next 7 days",
            "transfer lodestar rewards to niyant.eth on arbitrum",
            "swap all my plutus rewards to usdc",
            "wrap eth",
            "notify me when eth hits $3000",
            "unwrap weth into eth",
            "swap $100 ETH for USDC and bridge to Avalanche",
            "swap $100 of ETH to USDC and bridge to Avalanche",
            "Swap $100 of ETH to USDC on Arbitrum, and bridge to Avalanche",
            "swap 0.05 ETH for USDC on ARB and deposit on Lodestar ",
            "   swap 0.05 ETH for USDC on ARBITRUM and deposit on Lodestar",
            "Swap $100 ETH for USDC on ARBITRUM",
            "deposit all my USDC into Lodestar",
            "   deposit all my USDC into Lodestar",
            "swap all my usdc for ETH on Arbitrum ",
            "swap half my ETH for USDC on Arbitrum and deposit on Lodestar",
            "swap 0.05 ETH for USDC on Arbitrum and deposit on Lodestar",
            " deposit half my ETH into jonesdao",
            "deposit half my ETH on Arbitrum into jonesdao",
            "swap $100 of eth and $20 of wbtc into blur",
            "deposit 1 eth worth of usdc into dolomite",
            "transfer $50 eth and half my usdt to 7bfee.eth",
            "bridge half my usdc and all my dai to base",
            "bridge .01 eth to base chain when gas is less than $10",
            "swap all my usdt to avax",
            "what protocols are supported",
            "what actions are supported?",
            "whats current eth gas price",
            "which chains do you support?",
            "I want to stake my arb, please give me 3 options",
            "What I do with my stETH?",
            "Give me the 3 best DAI farms on ethereum",
            "What is the cheapest way for me to leverage up to go long on ETH and CRV",
            "Find profitable Arb opportunities on chain",
            "give me the highest yield strategy on ETH now and execute for me",
            "when ltv is greater than 80%, repay 10% of loan.",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 6:
        usermsgs = [
            "Lend all my usdc.e on arbitrum on Lodestar",
            "swap all of my usdc for usdc.e on arbitrum and lend all my usdc.e on lodestar",
            "swap all of my usdc for usdc.e and lend all usdc.e on lodestar",
            "bridge all ETH from ethereum mainnet to arbitrum",
            "swap 20 usdc.e for wbtc and lend wbtc on Lodestar hourly starting at 10:05 pm east",
            "swap 20 usdc.e for wbtc and lend wbtc on Lodestar every hour starting today at 10pm EAST.",
            "swap 20 usdc.e for wbtc and lend wbtc on Lodestar every hour starting at at 7PM EAST ",
            "swap 100 usdc for wbtc on arbitrum and lend wbtc on Lodestar",
            "swap all of my tokens on Arbitrum to ETH",
            "swap 100 usdc.e for usdc",
            "Buy wbtc with 100 usdc.e",
            "Buy WBTC with 100 USDC.E and Lend WBTC on Lodestar",
            "Buy WBTC with 100 USDC.E and Lend on Lodestar protocol",
            "swap 100 usdc.e for wbtc and lend all wbtc on Lodestar",
            "swap 100 usdc.e for wbtc and lend wbtc on Lodestar",
            "swap 100 usdc.e for 150 usdc and lend 95 on Lodestar",
            "swap 100 usdc.e for usdc and deposit 100 usdc.e on Lodestar",
            "lend 100 usdc.e on Lodestar",
            "deposit 100 usdc.e into Lodestar",
            "transfer 69 jones to 0x0B1A89664970EbEb16d6a1a039017049EEa45a20",
            "deposit 50 usdc in Rodeo GLP farm with 2x leverage",
            "depoist 50 usdc in pendle PT GLP 28MAR2024",
            "deposit 50 USDC in Pendle's GLP pool.",
            "swap 5 usdc.e for jones on arbitrum",
            "buy 2 jones with usdc.e",
            "buy 3 jones with usdc.e and deposit 100 usdc.e on Lodestar",
            "buy 3 jones with usdc.3 and deposit 100 usdc.e on Lodestar",
            "Supply 100 USDC.E on Lodestat",
            "buy jones with 3 USDC.E and deposit 75 USDC.E on Lodestar",
            "buy 3 jones with usdc.e ",
            "Buy 3 Jones with USDC.E and lend 75 USDC.E on Lodestar",
            "buy 3 Jones with USDC.E and lend 75 USDC.E on Lonestar",
            "buy 3 JONES with USDC.E and lend 75 USDC.E on Lonestar",
            "buy 50 USDC.E with ETH at 5:55PM pst",
            "Buy 50 USDC.E with ETH at 5:45PM PST",
            "Buy 50 USDC.E with ETH at 5:45PM PST",
            "Buy 50 USDC.E with ETH and deposit on Rodeo",
            "Buy 50 USDC.E and deposit into Silo",
            "swap 0.039 ETH for JONES on Arbitrum",
            "Buy 70 JONES with ETH",
            "buy 70 JONES with ETH",
            "bridge 0.01 eth from arbitrum to optimism",
            'bridge 0.04 ETH from Arbitrum to base and buy "Fren Pet"',
            "bridge 0.04 ETH from Arbitrum to base and buy Fren Pet",
            "bridge 0.04 ETH from Arbitrum to base and buy TOSHI",
            "buy boop with 0.01 ETH",
            "Bridge all of my ETH from Ethereum to Arbitrum. On Arbitrum, swap 0.05 ETH to USDC",
            "Bridge all of my tokens from Ethereum to Arbitrum",
            "bridge 0.01 ETH from Arbitrum to Optimism",
            "bridge 0.01 ETH from Arbitrum to Base",
            "bridge 0.069 ETH from Arbitrum to Base and buy USDC with 0.069 ETH on base",
            "bridge 0.069 wETH from Arbitrum to Base and buy USDC with 0.069 wETH on base",
            "bridge 0.069 ETH from Arbitrum to Base and buy USDC with 0.069 ETH",
            "bridge 0.069 ETH from Arbitrum to Base and buy USDC",
            "bridge 0.069 ETH from Arbitrum to Base and buy USDCb",
            "bridge 0.069 ETH from Arbitrum to base and buy USDC",
            "bridge 0.04 ETH from Arbitrum to base and buy USDC",
            "bridge 0.04 ETH from Arbitrum to base and buy TOSHI",
            "bridge 0.01 ETH from Arbitrum to base and buy TOSHI",
            "bridge 0.01 ETH to base and buy TOSHI",
            "bridge 0.01 ETH to base and buy 0xac1bd2486aaf3b5c0fc3fd868558b082a531b2b4",
            "bridge 0.01 ETH to base and buy 0xFF0C532FDB8Cd566Ae169C1CB157ff2Bdc83E105",
            "bridge 0.05 ETH to base and buy 0xFF0C532FDB8Cd566Ae169C1CB157ff2Bdc83E105",
            "bridge 0.05 ETH from Arbitrum to Metis, then buy  0x6Deb03fC15DA10BF25d542Eca0008d62463a7CBf on metis",
            "bridge 0.05 ETH from Arbitrum to Metis, then buy  0x6Deb03fC15DA10BF25d542Eca0008d62463a7CBf with 0.04 ETH",
            "bridge 0.05 ETH to Metis and buy 0x6Deb03fC15DA10BF25d542Eca0008d62463a7CBf",
            "buy 0x6Deb03fC15DA10BF25d542Eca0008d62463a7CBf on METIS",
            "Bridge 0.01 eth to base chain when gas is less than $10",
            "bridge all of my tokens from Canto to Ethereum",
            "bridge all of my WETH from Ethereum to Base using Hop",
            "bridge 1 ETH from Ethereum to Arbitrum using Bungee",
            "borrow 5000 USDT on Aave",
            "borrow 100 USDT on Rodeo",
            "borrow 3000 USDC on Lodestar",
            "transfer 0.0005 eth to 0xB6A5A72F5D811B6a6d9Ea653C919b6eAc6B1D3b2 and 0.0005 eth to 0xC59b5f658C16A6721A7f6D2bD1334aE8B53AD4dD",
            "transfer 0.0005 eth to 0xB6A5A72F5D811B6a6d9Ea653C919b6eAc6B1D3b2 when eth = $3000",
            "buy 0x8390a1DA07E376ef7aDd4Be859BA74Fb83aA02D5 with 0.01 ETH",
            "buuy 0x8390a1DA07E376ef7aDd4Be859BA74Fb83aA02D5 with 0.01 ETH",
            "Bridge my eth from scroll to eth mainnet via orbiter bridge",
            "Bridge all off my eth from scroll to eth mainnet via orbiter bridge",
            "deposit all my USDC into Lodestar ",
            "bridge eth to sepolia testnet",
            "Swap 0.15 ETH for Sushi",
            "Swap 0.05 ETH for SUSHI",
            "when gas is below 10 gwei, swap all my USDC to WETH",
            "when gas is below 10 gwei, swap my USDC to WETH",
            "claim uniswap airdrop",
            "claim airdrop",
            "payback loans on silo",
            "for the next two days, bridge .1 eth to arbitrum every 8 hours and then swap it for $USDC. ",
            "bridge .1 eth to arbitrum when gas is 20 or less and then swap it for $USDC. ",
            "bridge .1 eth to arbitrum when gas is 20 or less and then swap it for $USDC.e. Lend that $USDC.e on TimeSwap $USDC.e/$ARB pool",
            "bridge .1 eth to arbitrum when gas is 20 or less and then swap it for $USDC. Lend that $USDC on TimeSwap protocol pool",
            "bridge .1 eth to arbitrum when gas is 20 or less and then stake it at TimeSwap to farm $TIME token",
            "swap 2 usdc for eth when gas is above 20",
            "Swap all my usdc for eth when eth is $2022 usd or less",
            "Swap all my usdc for eth when eth is $2022 usd or less",
            "Swap all my usdc for eth when eth is $2022 usd",
            "Swap all my usdc for eth",
            "swap .03 eth for usdc",
            "swap .01 eth for usdc",
            "bridge .02 eth from mainnet to base chain and swap .01 eth for usdc, perform this when gas is less than 25",
            "Bridge .02 eth to base chain and swap .01 eth for usdc, perform this when gas is less than 20",
            "Bridge .02 eth to base chain and swap .01 eth for usdc, perform this when gwei is less than 25 ",
            "bridge .01 eth to base and swap it for usdc",
            "buy 50$ when BTC is at $37,790",
            "10x long doge",
            "deposit 0.05 ETH into the GENE-ETH pool on camelot",
            "claim $AAVE on Aave Protocol",
            "claime $AAVE",
            "swap 0.005 eth for usdc when eth = $3000",
            "when MKR trades at $1500, buy 1 MKR",
            "on ethereum mainnet buy 1 MKR",
            "claim from synthetix",
            "swap 0.0005 eth to usdc and transfer to 0xB6A5A72F5D811B6a6d9Ea653C919b6eAc6B1D3b2",
            "swap 0.0005 eth to usdc and then transfer to 0xB6A5A72F5D811B6a6d9Ea653C919b6eAc6B1D3b2",
            "transfer 0.0005 eth to 0xB6A5A72F5D811B6a6d9Ea653C919b6eAc6B1D3b2 then transfer to 0xa5Ef861278D7bf18A8A2068a01D66FBdeD93A1BD",
            "transfer 0.0005 eth to 0xB6A5A72F5D811B6a6d9Ea653C919b6eAc6B1D3b2 at 8pm est",
            "transfer 0.0005 eth to 0xB6A5A72F5D811B6a6d9Ea653C919b6eAc6B1D3b2 at 8pm est today",
            "swap 1 eth to usdc on Arbtirum",
            "swap 2 dai for eth in 1 minute",
            "swap 0,01 gmx for usdc on arbitrum chain",
            "swap 1 eth to usdc when marketcap is 2100",
            "bridge 100 usdc from arbitrum to ethereum and swap 10 eth to usdc",
            "bridge usdc from arbitrum to eth",
            "swap 1 eth to usdc when balance is 20 eth",
            "swap 1 eth to usdc when gas is 100 GWEI",
            "swap 1 eth to usdc when marketcap is 2500",
            "swap 1 eth to usdc at 11/20/2023",
            "swap 1 eth to usdc when 11/20/2023",
            "swap 1 eth usdc conditions eth price 1900$",
            "swap 1 eth usdc at eth price 1900$",
            "buy 10 $ARB with $ETH",
            "swap 10 $USCD for $ARB",
            "buy 10 $USDC using $ETH ",
            "buy $ETH with 10 $USDC on ARBITRUM ",
            "buy $ARB with 10 $USDC on ARBITRUM",
            "buy $ETH with 10 $USDC",
            "buy $ETH with 10 $USD ",
            "buy $ARB with 10 $USDC ",
            "swap $10 of USDC for ARB ",
            "swap 10$ USDC for ARB ",
            "buy 20 $USDC with ETH on Arbitrum",
            "swap $40 worth of ETH for $USDC on Arbritrum ",
            "can you withdraw my liquidity from ambient finance and bridge both tokens over to scroll please",
            "swap 0.02 eth to aura and deposit into jAURA vault on jonesdao",
            "deposit 20 usdc into the curve tri-pool",
            "swap 0.01 eth to usdc and then swap to usdt",
            "deposit 0.01 eth into aave, borrow 1 usdc",
            "deposit 0.01 eth into aave",
            "swap 0.01 eth for usdc at 1:05 pm est",
            "swap 0.01 eth for usdc in 1 minutes",
            "swap 0.01 eth for usdc in 3 minutes",
            "swap dai for et",
            "bridge all of my eth on arbitrum back to ethereum mainnet",
            "swap 100 eth to usdc on arbitrum and swap 100 eth to Canto on Cahtno",
            "swap 100 eth to usdt on ethereum and swap 100 eth to canto on Canto",
            "swap 100 eth to usdc on 1inch",
            "swap all of my gohm for eth and transfer it to 0xB6A5A72F5D811B6a6d9Ea653C919b6eAc6B1D3b2",
            "what are the possible values for 'protocolName' ?",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 7:
        usermsgs = [
            "deposit $50 worth of eth in aave",
            "in 3 minutes, bridge all my dai to arbitrum and swap to eth",
            "bridge 5 usdc from arbitrum to base using hop protocol",
            "swap usdc.e for 2 spa and deposit it into plutus on arbitrum",
            "deposit all of my $steth into the curve steth pool when the apy goes above 1.80%",
            "deposit all of my $steth into the curve steth pool on ethereum when the apy goes above 1.80%",
            "swap my toshi for 3 usdc on base, bridge it from base to arbitrum and sell it for arb on arbitrum",
            "swap 10 uni for woo on arbitrum in 1.1 hours",
            "swap all of my uni to grt on arbitrum at 9:30 PM GMT+2",
            "buy grail with 5 usdc.e on arbitrum at 18:00 GMT+8 in 2 days",
            "bridge 5 usdc.e from arbitrum to base in two days at this time",
            "swap usdc.e for 2 spa and deposit it all into plutus on arbitrum",
            "swap all of my $jones to usdc on arbitrum at 8:30 PM GMT+2 tomorrow",
            "swap eth for 2 usdc and deposit it into the gmx weth-usdc pool on arbitrum",
            "deposit 0.075 ETH into the MOZ-ETH vault on camelot",
            "swap all of my doge, shib and pepe to eth and deposit it into rocketpool on ethereum when gas is below 50",
            "sell mbs for bald when mbs market cap drops to $13,500,000 on base",
            "swap all of my doge, shib and pepe to eth and deposit it into rocket pool on ethereum when gas is sub 50",
            "swap all of my fxs and dpi on ethereum for blur",
            "buy $gmx with 0.01 eth and stake all of the $gmx on gmx on arbitrum",
            "when eth hits $2432 buy 50 usdc.e with eth on arbitrum and deposit on gmx",
            "buy 50 usdc.e with eth on arbitrum and deposit on gmx",
            "bridge 0.03 $eth from ethereum to arbitrum, buy $gmx with it, stake all of the $gmx on gmx on arbitrum",
            "deposit 20 $frax into the curve fraxusdp pool on ethereum when gas is below 52",
            "stake 10 eth on lido in exactly 3 days",
            "swap 15 dai for eth, swap 15 usdc for eth, bridge all of the eth from ethereum to arbitrum",
            "stake 0.015 eth on rocket pool on ethereum at noon tomorrow",
            "bridge all my DAI from ethereum to arbitrum and buy ARB when $ARB is below $2.12 and gas is sub 35",
            "when gas is above $500, swap 200 usdc to eth",
            "swap all of my jones for usdc on arbitrum, bridge it from arbitrum to base, and swap it for axl on base",
            "swap my dai and plsSPA to usdc on arbitrum",
            "repay my 3 MAGIC lodestar position on Arbitrum",
            "repay my 3 MAGIC lodestar loan on Arbitrum",
            "bridge 0.002 eth to zksync, swap it for usdc, then swap the usdc for eth, then swap the eth for usdc",
            "bridge 0.016 eth from arbitrum to zksync, swap 0.002 eth for usdc, then swap it for eth, then swap it for usdc",
            "buy grail with 5 usdc on arbitrum at 18:00 GMT+8 in 2 days",
            "swap all of my tokens on base to usdc and bridge it from base to arbitrum",
            "swap all of my FXS and DPI for BLUR",
            "deposit 0.001 eth and 5 usdc into the uniswap eth-usdc pool on ethereum",
            "Borrow 3 MAGIC on lodestar on arbitrum",
            "sell all of my $grail and $usdc.e for $usdc on arbitrum",
            "deposit 2 usdc into the gmx weth-usdc pool on arbitrum every thursday at 9pm utc",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 8:
        usermsgs = [
            "swap eth for 5 dai and lend it on aave on ethereum when dai supply apy goes above 9.86%",
            "swap all of my grt for usdc with 5% slippage on arbitrum, bridge it from arbitrum to base, and swap it for axl with 5% slippage on base",
            "swap 10 uni for woo on arbitrum in 1.1 hours",
            "swap 2 uni for wavax and deposit it into the gmx wavax-usdc pool on arbitrum",
            "swap 5 usdc for $shib on ethereum when $shib price goes below $0.0000096 and gas is sub 40",
            "deposit 0.001 eth and 5 usdc into the uniswapV3 eth-usdc pool on ethereum",
            "buy $gmx with 10 usdc and stake all of the $gmx on gmx on arbitrum",
            "deposit 0.003 eth into aave at 1:30pm est",
            "borrow 2 usdt from aave on ethereum, bridge it from ethereum to arbitrum and swap it for $joe on arbitrum",
            "buy 15 glp with usdc on arbitrum, deposit it into plutus, stake the plvglp on plutus",
            "deposit 2 usdc into the curve 3pool pool on ethereum when gas is sub 30",
            "withdraw all my gmx and repay it all on my borrow position on Lodestar on Arbitrum",
            "every monday at 7 PM EST, bridge from 0.1 ETH from Ethereum to Arbitrum using bungee. then bridge it back from Arbitrum to Ethereum using bungee",
            "bridge 0.005 eth from base to arbitrum using jumper",
            "transfer 0.005 eth on arbitrum to 0x28129f5B8b689EdcB7B581654266976aD77C719B",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 9:
        usermsgs = [
            "close all gmx positions",
            "Close arb short on gmx on arbitrum",
            "swap all my avax for usdt on trader joe",
            "stake all my eth on swell network",
            "deposit 30 usdc and the equivalent eth into ambient finance",
            "Buy ETH with 20 USDC every hour starting at 10:55PM",
            "Buy ETH with 20 USDC every hour starting at 10:55PM today",
            "repay my 0.0805 GMX borrow position on Lodestar on Arbitrum",
            "buy $50 worth of ETH at 2:25PM today.",
            "Swap usdc.e for 0.0805 GMX and repay my GMX on Lodestar",
            "Swap usdc.e for 0.085 GMX and repay my GMX borrow position on Lodestar on Arbitrum.",
            "what pools are supported on pendle?",
            "swap all of my tokens on base to ETH, then bridge eth to polygon",
            "swap eth for 2 matic on eth mainnet then bridge matic to polygon",
            "swap all DEGEN for ETH on Base",
            "repay all my GMX borrow position on Lodestar",
            "swap all of my tokens on BASE to ETH, transfer all ETH from Base to Polygon",
            "swap 0.05 eth to 0x8e16d46cb2da01cdd49601ec73d7b0344969ae33 on base",
            "send 0.00999 eth to 0x5881d9BfFf787C8655A9b7F3484aE1a6f7a966E8 on arbitrum",
            "swap all of my gmx for usdc on arbitrum",
            "swap max gmx to usdc on arbitrum",
            "withdraw all positions from lodestar",
            "what pools are supported on pendle ?",
            "what pools are supported on balancer ?",
            "Repay my GMX borrow position on Lodestar on Arbitrum",
            "in 45 minutes, bridge 600 usdc from optimism to ethereum and buy pepe",
            "swap all of my grt for usdc with 3% slippage on arbitrum, bridge it from arbitrum to base, and swap it for axl with 3% slippage on base",
            "swap my dai and plsSPA to usdc on arbitrum",
            "swap eth for 2 usdc and deposit it into the gmx weth-usdc pool on arbitrum at 9pm utc",
            "deposit 2 usdc into the gmx btc-usdc pool on arbitrum at 9pm est",
            "deposit 2 usdc into the curve 3pool pool on ethereum when gas is sub 30",
            "buy 3 usdc and 3 steur with eth on camelot and deposit both tokens in the camelot steur-usdc pool on arbitrum",
            "bridge 0.075 ETH from Arbitrum to Mantle and swap to USDY when ETH is below 2250",
            "swap 0.075 ETH to USDY when ETH is below 2250 and gas is below 35",
            "what protocols can i withdraw from?",
            "what protocols can i deposit into?",
            "what chains are supported",
            "Transfer 0.0001 ETH to 0x7886DaCE06ABdeb54929984CC9adeA78b42ed290 on Base every minute",
            "withdraw 1000 usdc.e from lodestar on arbitrum and buy 500 ROSNET",
            "on arbitrum, swap 20 usdc to grail on camelot at 22:00 utc daily",
            "swap half of degen for 0x0d97F261b1e88845184f678e2d1e7a98D9FD38dE on Base",
            "lend 0.15 ETH on lodestar on arbitrum, borrow 300 usdc from lodestar, open eth short with usdc on GMX v2 with 2.69x leverage",
            "swap all of my wETH for Frax on arbitrum, lend Frax on lodestar",
            "Transfer 5.4 matic to 0x5B7567Ed1bb7C338A20aF4eFb72e73dD6EF1dF61 on polygon",
            "swap all of my dai for uni via openocean on arbitrum",
            "what protocols are supported for lending on arbitrum ?",
            "deposit 0.034 eth and 100 usdc into the pancakeswap eth-usdc pool on base",
            "swap all of my tokens on base to USDC. bridge USDC from Base to Polygon. On polygon, swap USDC for YUP",
            "swap all of my tokens on base to ETH. bridge ETH from Base to Polygon. On polygon, swap ETH for YUP",
            "bridge ETH from ETH blockchain to ARB blockchain",
            "on arbitrum swap 100% of my gmx to eth, lend 0.01 eth to lodestar, and borrow 0.005 usdc from lodestar",
            "Swap 100 USDC to WBTC if BTC price falls under 60000",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 10:
        usermsgs = [
            "deposit 0.01 ETH into the sy-rETH pool on Arbitrum",
            "long doge on gmx",
            "swap bnb for eth",
            "transfer 5 bnb to 0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326",
            "transfer 12 arb to 0xD225cff23659a19996118ae544e9Dc0730D4bD31",
            "transfer 322 trump to 0x8C8D7C46219D9205f056f28fee5950aD564d7465",
            "swap 0x1f9840a85d5af5bf1d1762f925bdaddc4201f984 to 0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
            "swap 0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2 to 0x514910771af9ca656af840dff83e8264ecf986ca",
            "transfer 0xc00e94Cb662C3520282E6f5717214004A7f26888 to 0x70D8b972ef2A751f0dB12C0E67Dd21aE7B646797",
            "deposit 100 usdc into dai-usdc pool of aerodrome",
            "withdraw my ETH-GRAIL LP, swap half of my GRAIL to ETH and bridge to optimism when gas is below 25",
            "withdraw my ETH-GRAIL LP from Camelot, swap half of my GRAIL to ETH and bridge to optimism when gas is below 25",
            "withdraw my ETH-GRAIL LP from camelot",
            "withdraw 0xf82105aa473560cfbf8cbc6fd83db14eb4028117",
            "take 50% of the eth that I have and wrap it to weth",
            "wrap half the eth I have",
            "bridge 25 % of ETH to Optimism when gas is below 30 and buy UNI when UNI is below $13.12",
            "bridge 0.03 ETH to Optimism when it's cheaper than 3$",
            "bridge 0.03 ETH from arbitrum to base and buy CIRCLE when CIRCLE is below $15",
            "swap 0.025 ETH on BASE to CIRCLE when CIRCLE is below $15",
            "withdraw from all my camelot positions",
            "deposit 10 wbeth into sy-wbeth pool of pendle on bsc",
            "deposit 10 wbeth into sy-wbeth pool of pendle on bnb smart chain",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 11:
        usermsgs = [
            "stake 100 s*usdc into stargate",
            "stake 100 s*usdt into stargate",
            "deposit 1 eth on camelot and swap 1 eth for usdc",
            "swap 0.005 ETH on base to BASE",
            "use jumper on arbitrum and bridge .01 eth from arbitrum to base",
            "bridge all my usdt from bsc to base and swap it to eth on base",
            "swap half of 0xDFBb60D3245345aD8427C0a36a69E206e9DE8FA7 to ETH on BASE",
            "Deposit 4.955 PREMIA into sy-stargate-usdt pool on Pendle",
            "Bridge all of my MATIC from Polygon to BSC BNB",
            "bridge 9 USDT from BSC to Avalanche AVAX",
            "Bridge 50 USDT from BSC to Base ETH",
            "bridge all my usdt from bsc to base and swap it to eth on base",
            "swap 100 usdc to usdt on arbitrum and send 100 usdt to 0x0B1A89664970EbEb16d6a1a039017049EEa45a20",
            "when gas is below 0.15 long doge with 5x leverage with 310 usdc on gmx on arbitrum",
            "close doge position on gmx on arbitrum",
            "Withdraw 9 usdc from the usdc-arb pool on Compound ",
            "Lend 10 usdc on Compound on Arbitru ",
            "sell gmx on arbitrum to eth",
            "Deposit 100 USDC into the MUXLP pool on Pendle ",
            "Sell 20 usdc for ETH every hour starting at 10:45 pm today",
            "swap $100 worth of eth to usdc on sushi",
            "deposit 99 usdc and an equal amount of eth on uniswap on arbitrum",
            "bridge 100 usdc and 0.04 eth to arbitrum",
            "swap all of my tokens on base to USDC. bridge USDC from Base to Polygon. On polygon, swap USDC for YUP",
            "long doge with 5x leverage on gmx arbitrum using 0.05eth",
            "Swap 100 USDC to ETH if ETH price falls under 3100",
            "please swap my AERO on base to ETH",
            "swap all my gmx, lode, wbtc, and arb for eth on arbitrum",
            "bridge matic on polygon to usdt on optimism",
            "transfer 100 usdc on arbitrum to 0xe08583e015f358ce59489deba17b9774833c9f8e",
            "close my 3x leverage doge position on gmx on arbitrum",
            "close my 3x leverage doge long on gmx on arbitrum",
            "bridge arbitrum usdt to base usdc",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 12:
        usermsgs = [
            "buy eth with 100 usdc whenever eth goes below $3100 until tomorrow",
            "buy eth with 100 usdc whenever eth goes below $3100",
            "when gas is sub 21, swap $2000 worth of wbtc for eth",
            "swap 20$ of 0xd07379a755A8f11B57610154861D694b2A0f615a to eth on BASE",
            "swap 500$ of 0xd07379a755a8f11b57610154861d694b2a0f615a to ETH on BASE",
            "swap 5 polygon to eth on polygon",
            "swap all my BASE tokens to ETH on BASE and then bridhe these ETH on ETH. Then send all these ETH to this addy: 0xD896C7c5b9557e51c6339680bb9cab817299305C",
            "swap all of my grail to eth on arbitrum and bridge .005 eth to base",
            "swap eth to 50 USDC on arbtirum one",
            "bridge 100 USDC from base to arbitrum via jumper every monday and wednesday at 4 PM CET",
            "swap 100 USDC to ETH on arbitrum every monday and wednesday at 4:50 PM CET",
            "bridge eth from arbitrum to base every tuesday and thursday at 5 PM CET",
            "move 0.02 ETH from Ethereum to Base",
            "transfer 0.02 eth to my wallet on base",
            "Bridge $30 of ETH from Ethereum mainnet to Base L2 chain",
            "transfer 5 usdt from optimism to arbitrum",
            "deposit 1 eth and equal amount of usdc into aerodrome",
            "long btc with 5x leverage with 150 usdc on gmx and long eth with 5x leverage on gmx with 150 usdc and unwrap all my weth on arbitrum",
            "Using my arbitrum eth buy .01 worth of usdc",
            "bridge all my usdc from arb to base and bid aerodrome with it",
            "On Base swap TYBG for 0x010728385ce76c3f4f9ccb8b7f86cf49f6c56305",
            "swap 1.8 matic to usdc on polygon, bridge all usdc to base",
            "swap 1.5 matic to ETH on polygon then bridge to base",
            "swap all of my usdc on arbitrum for eth and then transfer 0.02 eth to base",
            "when btc is below falls below 70000 swap 250 USDC for ETH on Aribtum",
            "bridge my WETH on polygon to Manta network and swap it in ETH gas. Then send it to this address: 0x1C62C89cf3f57EAE7d61F1490C985ee82452752C",
            "swap every https://basescan.org/address/0xd07379a755a8f11b57610154861d694b2a0f615a to ETH on BASE",
            "on BASE blockchain, swap my 20$ of my BASE token to ETH",
            "swap 20$ of 0xd07379a755A8f11B57610154861D694b2A0f615a to eth on BASE",
            "swap my eth on arbtrim for 30 usdc and short sol with 15 with 2x on gmx",
            "swap 20$ of eth in usdc on jumper every tuesday for 2 weeks",
            "swap 1 eth to usdc when gas is below 10 until tomorrow",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 13:
        usermsgs = [
            "swap all matic to usdc on polygon then bridge all usdc from polygon to base",
            "swap blur on eth for pepe on blast",
            "swap all of my ondo on arb for wif on base",
            "swap all of my weth on base for eth on arbitrum",
            "swap 100 akt on optimism for stx on bsc",
            "swap ltc on polygon for 13 shib on optimism",
            "swap jasmy on base for 100 GALA on arbitrum",
            "swap bonk on op for floki",
            "swap eth for link on arb",
            "swap $100 of eth on arbitrum for $brett on base",
            "swap $100 worth of eth on avalanche for $brett on mainnet",
            "withdraw all my eth from lodestar. then long doge with 2x leverage",
            "swap all of USDC.e on polygon to matic via Ambient",
            "lend all of my usdc on arbitrum on lodestar. swap all of my DEGEN for ETH on base and bridge it to arbitrum. swap all of my WETH on base to eth.",
            "Close my sol long on gmx",
            "Close my sol position on gmx on arbitrum",
            "bridge 0.01 ETH from Arbitrum to Optimisim, Blast, and zkSych",
            "bridge 0.01 ETH from Arbitrum to Optimisim and to Blast",
            "Bridge 0.01 ETH to Optimism, Blast and Base",
            "Buy 0.05 eth worth of AERO at 18:00 cst",
            "wrap 0.02 eth",
            "unwrap 0.03 weth",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 14:
        usermsgs = [
            "swap $4.5k worth of wBTC for TRUMP on ethereum",
            "swap ETH to 100 USDC on base every tuesday and thursday at 6PM CET",
            "swap $670 of sol for wbtc for arbitrum",
            "buy 0.01 ETH of MAGA",
            "buy 0.01 ETH of PEPE",
            "buy 10 dollars worth of ETH of 0x576e2bed8f7b46d34016198911cdf9886f78bea7",
            "bridge all of my usdc across all chains to arbitrum",
            "claim all of my rewards across all chains from stargate and bridge then to arbitrum"
            "when my aave ltv hits 80%, deposit 200 usdc",
            "withdraw half my eth position from morpho when ltv goes below 40",
            "buy maga when fdv < 50 billion",
            "sell wld when fdv goes above 1,000,000,000,000",
            "when my lodestar health factor goes below 1.2, deposit all the usdt i have",
            "repay all my loans when morpho health factor goes below 1.1",
            "when compound health factor increases to 2, withdraw 1000 dai from it",
            "when gas doubles, sell all my eth",
            "deposit into morpho and bridge all my usdc to base when gas goes down 50%",
            "if eth price goes up 20%, buy PEPE with 30% of my usdc",
            "sell my DOGE when price 3x",
            "bridge to arbitrum and swap to gmx when gmx market cap dips 10%",
            "if market cap of $wif 3x, sell 10% of my wif",
            "if fdv of stg is below 400 million, unstake my stg",
            "deposit into the pendle yt-eth pool if pendle token fdv goes under 200000000",
            "when my DOG balance 5x, sell it all for eth",
            "buy wbtc with eth when my usdc balance decreases 50%",
            "deposit my eth and usdc into the uniswap pool with 3% range",
            "deposit my degen into the degen-eth pool on aerodrome with 5% range",
            "deposit 20 dai and 20 usdt into the curve 3pool",
            "withdraw from my camelot position, redeposit with 20% range",
            "deposit all my wbtc into the velodrome wbtc-eth pool with 10% range, then stake lp",
            "deposit 50 usdc and equivalent wbtc into 0xDef1C0ded9bec7F1a1670819833240f027b25EfF",
            "withdraw everything from 0x0fE7737956D706BC420f87B69AA4773CFc3b1A44",
            "withdraw my weth from 0x3cD751E6b0078Be393132286c442345e5DC49699 and deposit it into 0xF577628A1b2338f27e9331ea945c3b83f8DFd439",
            "buy eth with 100 usdc whenever eth goes below $3100 and buy usdc with 0.03 eth whenever eth goes above $3300 indefinitely",
            "whenever my balance for token x is greater than y (number) and the price of token x is greater than z (price), sell a (amount number) x (token name) for usdc",
            "deposit all eth into camelot eth-usdc pool and stake the lp",
            "buy eth with 100 usdc when eth is above 3000 or eth is below 1000",
            "if gas goes below 10 or gas decreases 50%, bridge all my wbtc to base and deposit into aerodrome",
            "Transfer 0.29 AVAX to 0x0e79459D098815bAB924Fb7a8a266448268B9650  on Avalanche by 1:34pm UTC",
            "swap .01 eth to usdc on arb",
            "bridge 0.03 ETH to Blast and swap 0.02 ETH to USDB on Thruster",
            "bridge 0.001 eth from arbritrum to base and swap half of the eth to usdc and deposit the eth and usdc to aerodrome on base",
            "lend .02 eth on dolomite on arbitrum, borrow 25 usdc from dolomite, and long eth with 2x leverage with the usdc on gmx",
            "buy BOOP with all of my usdc on arbitrum when BOOP market cap hits $17,800,000",
            "swap .02 eth to usdc on ethereum when gas goes below 7 and eth price hits $3000",
            "swap 84897.30038135161 of my token with contract 0xAC1Bd2486aAf3B5C0fc3Fd868558b082a531B2B4 to eth on base",
            "swap 0.0002 eth to Dia on base by 12am on uniswap",
            "Long ETH with 3x leverage with 10 USDC on gmx if ETH less than 3100 every 12 hours",
            "swap 0.01 eth to usdc on arbitrum one, try using 3% slippage and find me the best rate",
            "Add 10 usdc collateral to my btc position on gmx ",
            "bridge $100 of ETH from arbitrum to base on jumper every tuesday and thursday at 5 PM CET",
            "bridge $100 of ETH from arbitrum to base every tuesday and thursday at 5 PM CET",
            "bridge $100 of ETH from arbitrum to base on debridge every tuesday and thursday at 5 PM CET",
            "bridge $100 of ETH from arbitrum to base every tuesday and thursday at 5 PM CET. randomly select debridge or jumper as bridging solution",
            "bridge $0-$100 of ETH from arbitrum to base on jumper or debridge every tuesday and thursday at 5 PM CET",
            "bridge $0-$100 of ETH from arbitrum to base every tuesday and thursday at 5 PM CET. randomly select debridge or jumper as bridge solution",
            "bridge 0.01-0.02 of ETH from arbitrum to base on jumper or debridge every tuesday and thursday at 5 PM CET",
            "On arbitrum, Close arb position on gmx when arb is above $1.10",
            "on arbitrum, close ARB token short position on GMX, repay the USDC on lodestar and withdraw all lended ETH on lodestar",
            "withdraw all eth supplied to lodestar",
            "buy degen on base with 0.000001 ETH when market cap moves up or down 0.1% from current price",
            "buy eth on arbitrum with 1 usdc when price of eth drops 0.1%",
            "buy degen on base with 0.000001 eth when market cap drops 0.1%",
            "swap all usdc to eth on arbitrum when gas is low and eth price drops 0.1%",
            "bridge 0.01 eth from arbitrum to base and buy toshi with it at 3am today",
            "on base sell my degen to eth when profit in eth terms is 50%",
            "on base sell my degen to eth when profit in eth terms is 100%",
            "bridge 0.01 eth from arbitrum to base and buy degen with it at 2:30 pm eastern today",
            "bridge .01 ETH from arbitrum to base and transfer .019 eth to [0x66E751F8a564Be5b796e0E6d5d68FC7fa2c89976]",
            "swap all 0xAfb89a09D82FBDE58f18Ac6437B3fC81724e4dF6 for DEGEN on Base",
            "unlock eth and usdc on lodestar",
            "bridge all of my usdc on arbitrum to base and swap it to friend",
            "swap 0.005ETH to 0x576e2bed8f7b46d34016198911cdf9886f78bea7",
            "bridge 0.009ETH from base to arbitrum",
            "withdraw eth and usdc on lodestar",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)
    if step == 0 or step == 15:
        usermsgs = [
            "swap 0.01 eth to usdc on arbitrum, bridge to base and buy degen with it. then deposit in the degen eth pool on aerodrome",
            "Bridge all my eth on ethereum from ethereum to arbitrum eth",
            "bridge 5$ equivalent of eth on op  using bungee",
            "bridge 0.03 eth from arbitrum to blast then send 0.028 on blast to 0xc03E3DA10525B93f182C7D9FF4D6Aeca8A289bb4",
            "swap 19 000 000 meow on zksync to usdc",
            "close all my positions on gmx on arbitrum and swap it back to ethreum",
            "send all of my BNB on BSC to 0x10683d8452618CfCFEA3b918d17a58D09D5dB895 in 6 hours",
            "bridge 0.006 BNB from BSC to ETH on BASE",
            "bridge 0.006 BNB from BSC to BASE ETH",
            "bridge 0.02eth on arbitrum to base using debridge finance. then use 0.01 to buy $FLOPPA on base",
            "sell 0.15 eth on arbitrum for usdc, bridge to optimism, buy optimism and stake it to the tarot pool and then the topt in the tower pool",
            "Close my leveraged position on GMX",
            "bridge 15 usdc from base to polygon, buy matic with half, buy usdc with half",
            "bridge 0.01 eth from arbitrum to base and buy degen with 0.01 eth on base in 3 minutes",
            "buy degen with 0.005eth on base in the next 2 minutes",
            "swap all my available usdc back to eth on arbitrum",
            "bridge 0.009 eth from arbitrum to base using bungee and bridge back the same amount from base to arbitrum using jumper",
            "bridge all my ETH minus gas required from ethereum mainnet to arbitrum",
            "bridge 0.009 eth from arbitrum to base using bungee and bridge back the same amount from base to arbitrum using jumper at 12pm tommorow",
            "remove my eth position from aave on mainnet and deposit it on arbitrum pendle",
            "bridge 0.02eth on arbitrum to base using debridge finance. then use 0.01 to buy $FLOPPA on base",
            "bridge all of my USDC from BASE to BSC USDT",
            "Swap my virtual on base for eth",
            "Withdraw all collateral on compound",
            "Withdraw all ETH collateral on compound",
            "close short position on gmx",
            "repay my 20usdc loan on lodestar and withdraw 0.01eth being lent",
            "Repay all loans on compound",
            "Borrow  $3usdc on compound",
            "Deposit $9 of ETH on compound as collateral",
            "Use compound on arbitrum to borrow $3 usdc",
            "hello, please bridge all my eth on base to arbitrum and then swap 0.028eth to usdc",
            "repay 20usdc loan on lodestar",
            "long doge on gmx when funding rate goes above -0.13%",
            "short purr on hyperliquid when funding goes below 0.04",
            "close my ltc long on gmx if funding goes negative",
            "if open interest goes above 400m, short eth with all my usdc on hyperliquid",
            "if open interest goes down 20%, long btc on gmx",
            "close arb short on gmx if open interest increases 2x",
            "deposit weeth into pendle pt-weeth-27jun2024",
            "swap all my eth to rseth and deposit into the pendle yt-rseth-25sep2024 pool",
            "withdraw from pendle ezeth pool and deposit into the eeth pool",
            "withdraw from pendle rseth pool and deposit into the eeth-25sep2024 pool",
            "withdraw from aave and deposit into compound when compound apy is greater than aave",
            "withdraw from my pendle position when the apy is less than 10%",
            "when the pendle ezeth apy is greater than the pendle rseth apy, withdraw from my rseth position, swap to ezeth and deposit into the ezeth pool",
            "withdraw all of my weeth from the pt-weeth-25apr2024 pendle pool on arbitrum. then deposit it into the pt-weeth-26jun2024 pendle pool",
        ]
        for u in usermsgs:
            await process_message_ai(u.lower(), "", prompt, save, skip_db=True)


async def antisuite(prompt=1, save=False):
    """
    examples that dont work because we explicitly do not support them yet
    """
    _usermsgs = [
        "At Wednesday 11 PM UTC, vote for the pool with highest rewards in Chronos with APR greater than 150%",
        "Execute Stop losses (with specific parameters defined ofc)",
        "Add liquidity to asset x's LP pool in the y-z price range",
        "if an NFT of this collection gets listed 20% below floor price buy it and instantly list it at floor",
        "Hey Spice Bot, if price of ETH goes up and Health Factor on AAVE loan goes below 1.5 , borrow more USDC and buy more ETH if price hasn't increased more than 5% in past 24h, and make sure gas is below N before you do it all",
        "Leveraged looping on Aave",
        "When lock period expires on date_1 unbind LP and sell the lower of the two balances for the other staking resulting balance in vault",
        "whats my eth balance on all chains",
        "approve x token on uniswap router",
        "Use asset x as collateral for this long",
        "Uniswap LP management",
        "when a buy greater than X size takes place on tokenX sell %x of my position",
        "Every 1st and 15th of the month revoke all spending permissions on wallet1 where limit > 1.",
        "Short BTC at 38K with 50x leverage, long at 38k if 'X' twitter account tweets bearish",
        "buy with more usdc as the price goes lower, eg increase the buy usdc amount by 20% each 10% price drop",
        "If gas costs are below 10 and ETH-LONG liquidity on GMX is over 95% utilized, deposit 1000 USDC at 15x leverage",
        "Sell half of asset X as soon as price doubles from my entry",
        "On 30th OCT stack 100eth for 6months on Mantle ",
        "compose transaction that claims rewards from existing positions (where rewards are >$2), convert and/or stake reward to appropriate vault, and update bribes based on current allocation weight",
        "Discover the optimal yield for my ETH, including scenarios where its used as margin to borrow another higher-yield asset",
        "GLP farming. Like creating an algo to exit and enter at correct times",
        "Sell [X] NFT to the highest bidder on blur or opensea, bridge ETH to arbitrum and send to wallet: 0x",
        "Execute a perpetual basis spread trade focusing on funding rate arbitrage",
        "sell all tokens with balances under $100 and convert to USD",
        "get money from blur, send to arb, open up position on NFT perp all in one go",
        "Possessing ETH and desiring to short stETH against it, determine the most cost-effective approach to execute this on Mainnet (e.g., utilizing AAVE etc).",
        "Bridge and swap 0.2 ETH into required asset and lend into new Timeswap pool as soon as one launches",
        "Swaps all ETH from this address [address] into USDT and sends these USDT to this address [address]",
        "When <whale> makes a trade, automatically copy trade",
        "check defillama and give me everyweek a review of which protocols and which chains are gaining the biggest traction (in TVL inflow, DAU)",
        "make __(opensea, blur) bid at so so and so time",
        "Anytime between date_1 and date_2, if  funding rate >1% and ETH > 1600, short ETH using $1000 USDC",
        "If by date_1 farming rewards > 99$ then claim, else compound",
        "dump all my nfts",
        "Find the cheapest onchain put on ETH with strike price below 1234",
        "anytime this wallet buys, duplicate trade at x% of portfolio",
        "alert when (account address) buys for more than 1ETH",
        "Given my trust in USDT, USDC, DAI, and preference for Mainnet, Arbitrum, and Polygon chains, provide yield options available on AAVE, Silo.finance, MakerDAO, and Compound",
        "do a swap 10 usdt to zkusd on syncswap on all my accounts, at randomized time(random range is 8 hours)",
        "if btc price is above 27000 on a 4 hour close add 20% size to gmx v2 long",
        "Lending rate and borrowing rates for stable coins across different markets and chains are all different - thinks he would use this product to verify and quickly bridge, deposit, borrow, etc. to arb",
        "Buy 0.35 eth worth of $MOG and sell at a 2x",
        "whats my eth balance",
        "collateralize LST on polygon zkEVM, borrow DUSD, deposit DUSD in DUSD/USDC farms",
        "Identify and perform a spot DEX to on-chain perpetual arbitrage opportunity",
        "Engage in cross-chain yield arbitrage, such as borrowing ETH on AAVE Mainnet at 2% and lending it on AAVE Polygon for 10%",
        "when pool2 unlock threshold gets hit I want to remove my funds",
        "Leverage on-chain perpetuals like GMX and Gains Network for pair trading, while possibly utilizing spot or money markets like Aave to enter the most efficient short on SOL/DOT.",
        "pull my univ3 LP at price X and redeploy to range Y",
        "Tell me the cheapest way to to open a position on X based on what I have in my wallet",
        "Claim fSHROOMIEZ on Caviar as soon as reward > 1.1 and add back to Baton / Caviar LP with ETH counterpart, if/as soon as gas is below N",
        "If the pool's total volume decreases by x%, pull all of my funds in a frontrun",
        "Hey Spice Bot, if price of ETH goes up and Health Factor on AAVE loan goes below 1.5 , borrow more USDC and buy more ETH if price hasnt increased more than 5% in past 24h, and make sure gas is below N before you do it all",
        "claim my $tia airdrop for any wallet that is eligible in this set: addr1, addr2",
        "give me 10x leverage on the new crvUSD/sFRAX pools by looping my position accordingly",
        "when token x equals price y or less buy in increments of 0.5 eth no closer than every 15 minutes up to 5 eth using the cheapest aggregator (or maybe specify say 1inch) and then send to wallet b",
        "Sell all of my tokens under $100 and convert to USDT on mainnet",
        "make me a scanning of these addresses: XXX, XXX, and make a speadsheet with my different positions (name, money in $, time since I'm holding position, PNL on position, etc)",
        "Allocate collateral to Dopexs Atlantic ETH straddle, and acquire the most affordable hedge lasting for five Atlantic epochs (Note: Each epoch spans 2 days)",
        "Ok i want to 3x leverage GLP and zap out of USDC into some other asset thats support on Dolomite",
        "maintain healthy collateral ratio on synthetix. when gas is below 10, rebalance.",
        "Copytrade 0x123 with proportional position sizes if gas cost doesnt exceed 1/10 of position size",
        "send 0.2E 20 times to [wallet/contract address] today",
        "copy trade address X if token Y is greater than Z mcap with relative position size if it exceeds a certain threshold",
        "these 6 wallets on your monitoring list purchased XXX in amounts greater than XXX, I have constructed a trade that sells XX% of your ETH to purchase XXX of token XXX this is XX% of your portfolio / wallet balance.  Click here to execute",
        "Sell all tokens on the eth network below 10$ in 1 transaction",
        "the idea is youre minting DPXETH, you need USDC and RPDX and you need to bond and do a bunch of steps",
        "craft out the best route to get usdc from cosmos to avalanche for example",
        "check my positions on velodrome and aerodrome and 1 hour before the end of epoch tell me which pool should I vote for giving me the best potential APR",
        "sell all my CRV for Pendle on Arbitrum, lock it for 1 month, and when possible vote for the rETH pool",
        "Deposit ETH into Pendle when APY is 10%",
        "Whenever a Milady NFT (input ca) gets listed under 1.5eth, buy it",
        "Add liquidity to asset xs LP pool in the y-z price range",
        "Copytrade 0x123 with proportional position sizes if gas cost doesn’t exceed 1/10 of position size",
        "Compare and list onchain yield opportunities for NFT X or asset Y",
        "Utilize Opyns SQUEETH to hedge my position between ETH and USDC my position details currently are...",
        "When gas settles down consolidate my ETH to this address (would love it if its smart enough to detect relatively low gas vs having to put up an absolute number ",
        "help me snipe token A on block zero, if gas fee is above 30$ dont take the trade",
        "Track this wallet, spend $100 on the coin whenever this wallet makes a purchase >$100k on a coin",
        "arbitrage bot: buy btc on x and sell on y until price equalizes",
        "leverage long using curveusd with 10 ETH, close it once health drops below 1.1",
        "Swap usdc to usdt when price different is more than 1%",
        "When gas is below $20 on ETH Mainnet buy me 0.05ETH, bridge 0.03ETH under arb layer 2 and leave 0.01ETH in my mainnet account",
        "which ever token this wallet (provides the wallet to be tracked) buys with a minimum of 0.1eth, buy 0.1eth of it as well",
        "Provide me a snapshot of my current portfolio across x,y,z wallets and all chains with a token balance. Show me the % changes in weighting and value over the last month.",
        "sell all of my tokens under $100 and convert to USDT on mainnet",
        "whenever apr on this platform goes below APR on this other platform, liquidate first position to open it on this other platform",
        "Upon identifying a 2% price reduction (post-fees) for a Dopex $1600 ETH Call compared to a similar tenor, asset, and strike price option on Lyra, initiate a call parity trade by purchasing 10 Dopex calls and selling 10 corresponding calls on Lyra to capitalize on the spread/parity",
        "Claim YT-swETH yield on Pendle when gas fees are less than 5% of reward",
        "please monitor the APR on dexter, kujira blue and compare it to osmosis APR. Let me know when they offer better terms",
        "Wants to loop with pendle to earn yield; have to go pendle to buy PT, then go to a market to borrow PT, then go back to pendle to buy PT; i want my LTV at 60%” - loop until it gets there",
        "Send 1000 USDC to 0x2B605... when wallet value falls below $2000",
        "Borrow against my NFT on [Protocol X] at 60% LTV for 30 days, and stake the ETH on [Protocol Y]. 1 day before the loan is due, unstake and repay the loan",
        "Set A trade for 9am(New York time) Long position on eth using a 1000$ margin and close trade when 100 dollars is lost or when 150 is gain",
        "What tokens are releasing on this chain",
        "list all my nfts for sale on blur",
        "Provide liquidity to the most liquid pair on linea, zk sync and base",
        "Transfers 100 Coins staked on protocol X [dAPP link] to protocol Y [dAPP link]",
        "give me the best performing addresses that had the biggest combined PNL in % and the biggest combined PNL in $ of these token contracts XXX XXX",
        "when this address XXX has less than 0.05E send it 0.2E",
        "copytrade (account address) with 0.5 buy if he buys for  more than 1ETH",
        "Deposit $25 USDC in Satori and take a long position on ETH with a SL and TP of ... . Use the rest of the USDCs to reach at least $200 trading volume using USDC/USDT pair",
        "Deposit 1000 USDC and borrow ARB on Paraspace, bridge from Arbitrum to Ethereum, deposit ARB and borrow USDC on Compound, deposit USDC on GMD Protocol",
        "I want to farm USDC/DAI LP. Where do I get the best yields for this liquidity position and how do I enter the position",
        "if gas costs are below 10 and ETH-LONG liquidity on GMX is over 95% utilized, deposit 1000 USDC at 15x leverage",
        "Calculate the $ value of slippage for executing an  <insert size> swap for <token pair / liquidity pool>",
        "Remove lent assets from Timeswap pool at maturity (and notify me)",
        "If liquidity withdrawn in pool X is above 10M, withdraw asset A",
        "Employ spot and money markets like AAVE and Silo to execute pair trades, for instance, lending ETH, borrowing BTC, and selling the borrowed BTC for ETH.",
        "Update the range on my Uniswap position to the recommended values",
        "Remove lent assets from Timeswap pool at maturity",
        "create lp pair rdpx/eth and compund rewards every week.",
        "Send one Hedgies NFT to [inputs address]",
        "Sell my XYZ as soon as 0x123 sells",
        "On 30th OCT stack 100eth for 6months on Mantle",
        "Monitoring uni v3 lp position and constantly balance it to keep it in a certain range or adjusting the range",
        "Transfers all ETH at this address [address] from the arbitrum blockchain to the ether blockchain.",
        "Long/short X if specific influencer tweets a ticker",
        "Recommend me some pools to farm yield with above 10% APY using my ETH. Parameters include - farm TVL >$5mil, rewards are majority in ETH and not native tokens.",
    ]


def standardize():
    blacklist = [
        "withdraw eth and usdc on lodestar",
        "claim livethe rewards once they are worth greater than 10$ and compound them back into livethe",
        "carry out x no of swap on the dapp  daily for 1 month when gas is less than 30"
        "harvesting on camelot",
        "send $1000 to [insert wallet] every",
        'at exactly 10pm tomorrow buy "random ca" with 40 gwei',
        'swap 0.05 eth to usdt and send the swapped usdt to "wallet address"',
        "borrow $400 usdc and swap to $bitcoin",
        "swap asset x for y, at % max price impact",
        "vote for the highest yielding bribe on thena on wednesday at 2355 utc",
        "buy xxx$ of $ticker with stablecoins/from my usdc",
        "disperse eth into few wallets",
        "deposit usdc and eth into camelot usdc/eth pool, stake the spnft into a nitro pool",
        "using 2 eth buy usdc, usdt, and dai, then deposit into curve tricrypto pool",
        "grab weekly rewards from ve(3,3) dexes and convert them to eth",
        "lend eth as collateral on wing-finance which earns a supply apy of 90.79%. borrow usdt against your eth collateral with a max ltv of 85% and a borrow apy of -1.55% (the interest you need to pay). farm with usdt on paraspace-lending-v1 which earns 26.65%.",
        "claim rlbtrfly rewards whenever available, convert to eth and deposit into the spice blur vault",
        "perform $xxx swap to $yyy using ___ (specific dex) when gas is below __(given gwei)",
        "bridge through stargate 100 usdc with cheapest fee's on any eth l2",
        "bridge 1 eth for usdc with 2% slippage",
        "bridge 1 eth for usdc then swap to dai with max 2% slippage",
        "swap 0.25eth for *contract-adress using 5% slippage",
        "deposit 0.35 eth into aave, borrow $400 usdc and swap to $bitcoin",
        "at exactly 19:05 utc, bridge $50 eth to starknet eth",
        "claim rewards from tig staking and tigusd staking and then add that tigusd back to tigusd position in tigristrade everyday if the tigusd rewards balance is above 5",
        "swap asset # x for y, at % max price impact",
        "claim stg from my stargate positions, swap to weth, and deposit back into stargate",
        "swap by 2 $eth for $geth, convert to 1 $reth and 1 $steth, stake both on rocketpool",
        "give me an alert whenever stablecoin inflow see a significant growth over 30d period (atleast +10%)",
        "stake my arb on arbitrum",
        "bridge eth to arb when gas is sub .5$ and swap back when arb hits .90",
        "sell everything from multiple chains into jusdc",
        "give me the highest yield strategy on eth now and execute for me",
        "deposit usdc into jones jusdc and deposit to rodeo finance usdc: (contract address)",
        "loop 1 eth with usdc on lodestar 3 times",
        "buy pt with usdc on pendle, then loop pt on dolomite 5 times",
        "once eth balance hits x, buy y of $token when gas",
        "buy 3 jones with usdc.3 and deposit 100 usdc.e on lodestar",
        "on gmx v2, when eth price is near x, can you deposit x usdc collateral to avoid liquidation",
        "each day, claim my arb rewards on jones dao from the jglp & jusdc masterchef contract and convert to $eth",
        "pull my liquidity on uniswap if price falls below x",
        "claim blur points whenever they release, swap them to eth if gas is <40",
        "whenever i have over $4000 in ethereum, send $3500 to the gemini address, but only do this once a week at most",
        "once eth balance hits x, buy y of $token when gas is below z",
        "when gas is below x use defillama to swap eth or usdc to x coin",
        "sell x% of x$coin when price is x",
        "swap my $bitcoin on the dex with the least amount of slippage",
        "sell x% of y coin (or insert contract address?) every (day of week) at 12am utc on (insert preferred dex).",
        "exchange all existing tokens in my wallet for eth. once, finished send it to [cex deposit wallet]",
        "can you gather all the eth that i have spread around different chains and bridge it in the most cost-efficient form to arbitrum",
        "transfer 100 coins staked on protocol x [dapp link] to protocol y [dapp link].",
        "depoist 50 usdc in pendle pt glp 28mar2024",
        "bridge eth to sepolia testnet",
        "craft out the best route to get usdc from cosmos to avalanche",
        "bridge [amount] eth from ethereum to arbitrum using the most cost-effective method. then, convert it to weth.",
        "i have an existing dola usdc position on aurat. can you help me harvest my yield every week and compound into the same position?",
        "claim rewards from thena/velodrome/aerodrome, convert them to eth and bridge them to ethereum",
        "swap 100 usdc.e for 150 usdc and lend 95 on lodestar",
        "bridge 0.04 eth from arbitrum to base and buy fren pet",
        "bridge 3.2 ethereums to mainnet",
        "bridge all eth at this address [address] from the arbitrum blockchain to the ether blockchain",
        "on the 16 october bridge 0.5 eth to zk sync using (app name) when gas is below 20 gwei",
        "claim and sell and stg rewards",
        "send/transfer 1 eth on base and give me usdc on zksync",
        "swap eth for usdt, swap usdc for usdt, bridge usdt to arbitrum",
        "claim rewards from camelot, swap rewards and grail into xgrail, then deposit xgrail into camelot",
        "Lend WETH as collateral on polylend, borrow WETH on polylend",
        "bridge 500 usdc  each to linea, zk sync and base when gas is below 10",
        "swap usdc for arb if the price reaches $0.90",
        "disperse 0.1e to these 10 wallets",
        "deposit 50 usdc in rodeo glp farm with 2x leverage",
        "harvest on sushi and sell/restake.",
        "swap 1 eth to usdc when 11/20/2023",
        "unvest 300000 trove from nitro cartel and swap to usdc; revest remaining trove at 16:00 utc",
        "if ldo goes down 10% or has a onchain liquidation bid it",
        "set limit orders to buy eth each time the price dips 10% from the current price, buy for 100 usdc each time.",
        "bridge 20$ from polygon to arbitrum",
        "vote for the most profitable strategy on any >10m mcap -pool in the thena ve(3,3) voting pools. do this at 11:55. at 12:05, collect the rewards of the previous voting epoch and exchange them for doge at market prices, using swap aggregator swap.defillama.com with 0.5% slippage.",
        "bridge 500 usdc each to linea, zk sync and base when gas is below 10",
        "what are the possible values for 'protocolname' ?",
        "loop 1 eth with usdc on lodestar 3 times",
        "buy pt with usdc on pendle, then loop pt on dolomite 5 times",
        "swap 1 eth usdc at eth price 1900$",
        "deposit all of my wsteth in the kyber axl-wsteth-wsteth pool on pendle",
        "swap all eth from this address [address] into usdt and sends these usdt to this address [address]",
        "bridge .1 eth to arbitrum when gas is 20 or less and then stake it at timeswap to farm $time token",
        "bridge 2 $eth to arbitrum, swap $eth to $gmx, open an 0.5 $eth short position with 10x leverage with $eth market price, set stop loss at $1500 per $eth",
        "cheapest route for eth to arbitrum now",
        "bridge 100 of x on base to bnb and swap to y",
        "buy 0.35 eth worth of $mog and sell when mog/eth hits 0.7",
        "deposit 1 gril into camlot",
        "buy ustd with 2 eth",
        "set stop loss for eth on arbitrum chain. sell 0.1 eth when price goes lower than 1500",
    ]
    keywords = get_keywords()
    protocols = get_protocols()
    chains = get_chains()
    shortened_chains = get_shortened_chains()
    tokens = get_tokens(keywords)
    with open("test/preprocessed.json", "r") as f:
        mesgs = json.load(f)
    with open("test/newanswers.json", "r") as f:
        data = json.load(f)
    with open("verified_entities.json", "r") as f:
        ve = json.load(f)
    if ve:
        for p in ve["protocols"]:
            if "name" in p:
                protocols.add(p["name"].lower())
        for c in ve["chains"]:
            if "name" in c:
                chains.add(c["name"].lower())
            if "tokens" in c:
                for t in c["tokens"]:
                    if (
                        t["symbol"] not in keywords
                        and t["symbol"] != ""
                        and not any(
                            z in t["symbol"]
                            for z in [
                                ".",
                                "+",
                                "*",
                                "?",
                                "^",
                                "$",
                                "(",
                                ")",
                                "[",
                                "]",
                                "{",
                                "}",
                                "|",
                                "\\",
                            ]
                        )
                        and not all(zz.isdigit() for zz in t["symbol"])
                    ):
                        tokens.append(t["symbol"].lower())
                    if (
                        t["name"] not in keywords
                        and t["name"] != ""
                        and not any(
                            z in t["name"]
                            for z in [
                                ".",
                                "+",
                                "*",
                                "?",
                                "^",
                                "$",
                                "(",
                                ")",
                                "[",
                                "]",
                                "{",
                                "}",
                                "|",
                                "\\",
                            ]
                        )
                        and not all(zz.isdigit() for zz in t["name"])
                    ):
                        tokens.append(t["name"].lower())
    tokens = set(tokens)
    rxtk = r"(?P<tkn>" + r"|".join(tokens - protocols - keywords) + r")"
    rxtk2 = r"(?P<tkn2>" + r"|".join(tokens - protocols - keywords) + r")"

    # standardize
    new = {}
    count = 1
    harmless = []
    different = []
    for k, va in list(data.items()):
        if k in blacklist:
            new[k] = va
            continue
        # print(k, v)
        newv = []
        for iy, v in enumerate(va):
            if v == []:
                continue
            oldv = json.dumps(v)
            message = mesgs[k]["new"]
            updated = mesgs[k]["updated"]
            results = []
            for vv in v:
                results.append(
                    {"name": vv["name"], "arguments": json.dumps(vv["args"])}
                )
            signs, _ = processing(
                results,
                None,
                message,
                updated,
                chains,
                protocols,
                tokens,
                shortened_chains,
                rxtk,
                rxtk2,
            )
            nv = postprocessing(
                signs, message, k, protocols, chains, tokens, "ethereum"
            )
            if nv != json.loads(oldv):  # and iy == 0:
                for nnv in nv:
                    if (
                        "inputAmount" in nnv["args"]
                        and isinstance(nnv["args"]["inputAmount"], list)
                        and "inputToken" in nnv["args"]
                        and isinstance(nnv["args"]["inputToken"], list)
                    ):
                        # print(k)
                        harmless.append(k)
                        break
                else:
                    print(count, k, nv)
                    count += 1
                    different.append(k)
            if k in harmless or k in different:
                newv.append(json.loads(oldv))
            else:
                newv.append(nv)
        new[k] = newv

    with open("test/newanswers.json", "w") as f:
        json.dump(new, f)

    # save
    new2 = {}
    for k, va in list(new.items()):
        if k in blacklist or k in different:
            continue
        skip = False
        v = va[0]
        for w in v:
            if w["name"] in [
                "vote",
                "chat",
                "support",
                "positions",
            ]:
                skip = True
                break
        if skip:
            continue
        new2[k] = v

    with open("test/full.json", "w") as f:
        json.dump(new2, f)

    # check
    for k, v in list(new2.items()):
        for ix, i in enumerate(v):
            for a, b in list(i["args"].items()):
                if b == "outputAmount" or b == "outputToken":
                    if ix > 0:
                        if v[ix - 1]["name"] == "condition":
                            print("a", k)
                            break
    for k, v in list(new2.items()):
        for ix, i in enumerate(v):
            for a, b in list(i["args"].items()):
                if (
                    a in ["token", "inputToken", "amount", "inputAmount"]
                    and b == "all"
                    and "all" not in k
                    and "everything" not in k
                ):
                    print("b", k)
                    break
    for k, v in list(new2.items()):
        for ix, i in enumerate(v):
            for a, b in list(i["args"].items()):
                if b == "outputToken":
                    if ix > 0:
                        if "outputToken" in list(v[ix - 1]["args"].keys()):
                            print("c", k)
                            break


def split_data(data, train_ratio=0.66, val_ratio=0.17):
    keys = list(data.keys())
    total_length = len(keys)
    print(total_length)
    train_size = int(total_length * train_ratio)
    val_size = int(total_length * val_ratio)

    # Shuffle the data
    random.shuffle(keys)

    # Split the data
    train_data = {k: data[k] for k in keys[:train_size]}
    val_data = {k: data[k] for k in keys[train_size : train_size + val_size]}
    test_data = {k: data[k] for k in keys[train_size + val_size :]}

    important = [
        m.lower()
        for m in [
            "withdraw from aave and deposit into compound when compound apy is greater than aave",
            "withdraw from my pendle position when the apy is less than 10%",
            "when the pendle ezeth apy is greater than the pendle rseth apy, withdraw from my rseth position, swap to ezeth and deposit into the ezeth pool",
            "bridge 0.006 BNB from BSC to ETH on BASE",
            "bridge 0.006 BNB from BSC to BASE ETH",
            "bridge 0.02eth on arbitrum to base using debridge finance. then use 0.01 to buy $FLOPPA on base",
            "bridge all off my eth from scroll to eth mainnet via orbiter bridge",
            "buy 10 dollars worth of eth of 0x576e2bed8f7b46d34016198911cdf9886f78bea7",
            "when eth is below $1600, buy $500 usd worth each week",
            "repay all my loans when morpho health factor goes below 1.1",
            "trade 1 baby bear for eth then swap that eth for usdc and send that usdc over to the arbitrum network with hop exchange",
            "buy eth with 100 usdc whenever eth goes below $3100 until tomorrow",
            "when my lodestar health factor goes below 1.2, deposit all the usdt i have",
            "if token x goes -30%, sell to usdc",
            "bridge $100 of eth from arbitrum to base every tuesday and thursday at 5 pm cet. randomly select debridge or jumper as bridging solution",
            "swap eth to 100 usdc on base every tuesday and thursday at 6pm cet",
            "bridge all of my usdc across all chains to arbitrum",
            "withdraw half my eth position from morpho when ltv goes below 40",
            "when gas doubles, sell all my eth",
            "sell my doge when price 3x",
            "if market cap of $wif 3x, sell 10% of my wif",
            "buy wbtc with eth when my usdc balance decreases 50%",
            "deposit my eth and usdc into the uniswap pool with 3% range",
            "deposit my degen into the degen-eth pool on aerodrome with 5% range",
            "withdraw from my camelot position, redeposit with 20% range",
            "deposit all my wbtc into the velodrome wbtc-eth pool with 10% range, then stake lp",
            "deposit 50 usdc and equivalent wbtc into 0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            "withdraw everything from 0x0fe7737956d706bc420f87b69aa4773cfc3b1a44",
            "withdraw my weth from 0x3cd751e6b0078be393132286c442345e5dc49699 and deposit it into 0xf577628a1b2338f27e9331ea945c3b83f8dfd439",
            "buy eth with 100 usdc whenever eth goes below $3100 and buy usdc with 0.03 eth whenever eth goes above $3300 indefinitely",
            "whenever my balance for token x is greater than y (number) and the price of token x is greater than z (price), sell a (amount number) x (token name) for usdc",
            "if gas goes below 10 or gas decreases 50%, bridge all my wbtc to base and deposit into aerodrome",
            "add 10 usdc collateral to my btc position on gmx ",
            "bridge $0-$100 of eth from arbitrum to base every tuesday and thursday at 5 pm cet. randomly select debridge or jumper as bridge solution",
            "bridge $0-$100 of eth from arbitrum to base on jumper or debridge every tuesday and thursday at 5 pm cet",
            "bridge 0.01-0.02 of eth from arbitrum to base on jumper or debridge every tuesday and thursday at 5 pm cet",
            "on base sell my degen to eth when profit in eth terms is 50%",
            "on base sell my degen to eth when profit in eth terms is 100%",
            "buy 1 eth with usdt if price goes to 1200$",
            "bridge 0.01 eth from arbitrum to optimisim and to blast",
            "bridge 0.01 eth to optimism, blast and base",
            "swap all of my weth on base for eth on arbitrum",
            "swap blur on eth for pepe on blast",
            "swap all of my ondo on arb for wif on base",
            "swap 100 akt on optimism for stx on bsc",
            "swap ltc on polygon for 13 shib on optimism",
            "swap jasmy on base for 100 gala on arbitrum",
            "swap bonk on op for floki",
            "swap eth for link on arb",
            "swap $100 of eth on arbitrum for $brett on base",
            "swap $100 worth of eth on avalanche for $brett on mainnet",
            "bridge half my usdc and all my dai to base",
            "swap $100 of eth and $20 of wbtc into blur",
            "on every wednesday at 10am until the end of november send 500usdt to address 0x-------- and $500 worth of eth to 0x---- when gas is below 10",
            "swap all my wbtc for usdt at 12pm tomorrow or if usdt price goes below $0.9",
            "buy eth with 5000 usdc. sell eth for usdc if the price goes below 1000 or above 3000",
            "lock steth for 2 months",
            "swap 1 $eth to $usdc and then bridge it to arbitrum",
            "buy btc with 1 eth when btc is at or below $25000 and sell 0.2 btc for eth when btc is at or above $30000, forever",
            "swap all my wbtc for usdt when my eth balance is greater than 2 or eth/dai goes above 2000",
            "swap all my merit circle, dai, fxs, for eth. then brdige all the eth as well as my 12,227 usdc position over to arbitrum when gas is <10",
            "swap all my usdt and usdc for dai",
            "withdraw all my usdc and usdt from rodeo, convert to eth, and bridge all of it to mainnet",
            "for the next two days, bridge .1 eth to arbitrum every 8 hours and then swap it for $usdc. ",
            "send 0.01e to [wallet/contract address] every hour for 5 days",
            "bridge my eth from scroll to eth mainnet via orbiter bridge",
            "bridge 1 ether to arbitrum via hop",
            "deposit $50 worth of eth in aave",
            "in 3 minutes, bridge all my dai to arbitrum and swap to eth",
            "swap usdc.e for 2 spa and deposit it into plutus on arbitrum",
            "deposit all of my $steth into the curve steth pool when the apy goes above 1.80%",
            "deposit all of my $steth into the curve steth pool on ethereum when the apy goes above 1.80%",
            "buy grail with 5 usdc.e on arbitrum at 18:00 gmt+8 in 2 days",
            "swap usdc.e for 2 spa and deposit it all into plutus on arbitrum",
            "bridge 0.03 $eth from ethereum to arbitrum, buy $gmx with it, stake all of the $gmx on gmx on arbitrum",
            "use jumper on arbitrum and bridge .01 eth from arbitrum to base",
            "buy $50 worth of eth at 2:25pm today.",
            "buy 5,000 usdc worth of eth and then bridge it to zk sync era using orbiter bridge. then at noon each day for the next week swap $500 of eth for usdc and then swap back $500 worth of usdc for eth",
            "when $tok is at 12k market cap, buy $20 worth of $tok",
            "swap 100 usdc.e to eth on llamaswap and long eth with 5x leverage on gmx when eth is $1550",
            "if eth hits 1200 usd open a 12x long on gmx",
            "when gas is below 0.15 long doge with 5x leverage with 310 usdc on gmx on arbitrum",
            "open a short trade on kwenta on btc with 3 eth with 3x leverage",
            "short eth with usdc if it goes below or touches x price",
            "close my 3x leverage doge position on gmx on arbitrum",
            "buy wbtc with 100 usdc.e and lend wbtc on lodestar",
            "buy 50 usdc.e and deposit into silo",
            "swap eth for 5 dai and lend it on aave on ethereum when dai supply apy goes above 9.86%",
            "bridge 50 usdt from bsc to base eth",
            "bridge 9 usdt from bsc to avalanche avax",
            "bridge all of my matic from polygon to bsc bnb",
            "bridge matic on polygon to usdt on optimism",
            "swap $100 of eth to usdc and bridge to avalanche",
            "withdraw 1000 usdc.e from lodestar on arbitrum and buy 500 rosnet",
            "buy 50 usdc.e with eth and deposit on rodeo",
            "short eth if it goes below or touches xxx price",
            "swap my eth on arbtrim for 30 usdc and short sol with 15 with 2x on gmx",
            "swap 20$ of eth in usdc on jumper every tuesday for 2 weeks",
            "swap 1 eth to usdc when gas is below 10 until tomorrow",
        ]
    ]
    important_train = int(len(important) * (train_ratio + val_ratio))
    random.shuffle(important)
    # Split the data
    forcetrain = important[:important_train]
    forceval = important[important_train:]
    for fv in forceval:
        if fv in val_data:
            continue
        elif fv in train_data:
            val_data[fv] = train_data[fv]
            del train_data[fv]
        elif fv in test_data:
            val_data[fv] = test_data[fv]
            del test_data[fv]
        else:
            raise Exception(f"what? {fv}")
    for ft in forcetrain:
        if ft in train_data:
            continue
        elif ft in val_data:
            train_data[ft] = val_data[ft]
            del val_data[ft]
        elif ft in test_data:
            train_data[ft] = test_data[ft]
            del test_data[ft]
        else:
            raise Exception(f"what? {ft}")

    print(len(train_data), len(val_data), len(test_data))

    return train_data, val_data, test_data


def split():
    # Read data from the original JSON file
    with open("test/full.json", "r") as file:
        original_data = json.load(file)

    # Split the data
    train_data, val_data, test_data = split_data(original_data)

    # Save the split data into three separate JSON files
    with open("test/traindata.json", "w") as f:
        json.dump(train_data, f)
    with open("test/valdata.json", "w") as f:
        json.dump(val_data, f)
    with open("test/testdata.json", "w") as f:
        json.dump(test_data, f)


async def create_ft_file(infile, outfile) -> Any:
    print("reading", infile, "and writing to", outfile)
    available_functions = get_available_functions()
    with open("test/preprocessed.json", "r") as f:
        mesgs = json.load(f)
    with open(infile, "r") as f:
        data = json.load(f)
    ds = list(data.keys())
    with open("tools-min.json", "r") as ifile:
        fns = [tool["function"] for tool in json.load(ifile)]
    ofile = open(outfile, "w")
    for query in ds:
        message = mesgs[query]["new"]
        fnames = []
        msgs: list[dict[str, Any]] = [{"role": "user", "content": message}]
        for fn in data[query]:
            fname = fn["name"]
            fargs = fn["args"]
            msgs += [
                {
                    "role": "assistant",
                    "function_call": {"name": fname, "arguments": json.dumps(fargs)},
                }
            ]
            fcall = available_functions[fname]
            fresponse = await chat(**fargs) if fname == "chat" else fcall(**fargs)
            msgs += [{"role": "function", "name": fn["name"], "content": fresponse}]
            fnames.append(fname)
        if fnames[-1] not in ["chat", "support", "positions"]:
            finalmsg = "Completed with calls to"
            for fnme in fnames:
                finalmsg += f" {fnme},"
            finalmsg = finalmsg[:-1]
            msgs += [{"role": "assistant", "function_call": None, "content": finalmsg}]
        else:
            msgs = msgs[:-1]
        line = {"messages": msgs, "functions": fns}
        ofile.write(json.dumps(line) + "\n")
    ofile.close()


async def generate() -> Any:
    infile = "test/traindata.json"
    outfile = "test/traindata.jsonl"
    await create_ft_file(infile, outfile)
    trainfile = await client.files.create(file=open(outfile, "rb"), purpose="fine-tune")
    infile = "test/valdata.json"
    outfile = "test/valdata.jsonl"
    await create_ft_file(infile, outfile)
    valfile = await client.files.create(file=open(outfile, "rb"), purpose="fine-tune")
    time.sleep(12)
    print(trainfile.id, valfile.id)
    job = await client.fine_tuning.jobs.create(
        training_file=trainfile.id,
        validation_file=valfile.id,
        model="gpt-3.5-turbo-0125",
        hyperparameters={"n_epochs": 1},
        # using auto batch size and learning rate hyperparameters
    )
    print(job.id)


async def run_preprocess():
    with open("test/newanswers.json", "r") as f:
        data = json.load(f)
    with open("test/preprocessed.json", "r") as f:
        have = json.load(f)
    (
        actions,
        protocols,
        tokens,
        chains,
        shortened_chains,
        keywords,
        pools,
        orderedtokens,
        confusing1,
        confusing2,
        confusing3,
    ) = await entities(fast=False)
    rxtk = r"(?P<tkn>" + r"|".join(tokens - protocols - keywords) + r")"
    rxtk2 = r"(?P<tkn2>" + r"|".join(tokens - protocols - keywords) + r")"
    rxact = r"(?P<act>" + r"|".join(actions) + r")"
    rxch = r"(?P<chn>" + r"|".join(chains) + r")"
    new = {}
    count = 0
    print(len(list(data.keys())))
    for k in list(data.keys()):
        if k in list(have.keys()):
            new[k] = {"new": have[k]["new"], "updated": have[k]["updated"]}
            # continue
        # print(k)
        (
            om,
            message,
            updated,
            error_messages,
        ) = await preprocessing(
            k,
            rxtk,
            rxtk2,
            rxact,
            rxch,
            keywords,
            tokens,
            protocols,
            chains,
            shortened_chains,
            pools,
            orderedtokens,
            actions,
            confusing1,
            confusing2,
            confusing3,
        )
        if k in new and message != new[k]["new"]:
            print(f"{k}, {new[k]['new']}, {message}")
        del updated["actions"]
        del updated["orderedtokens"]
        new[k] = {"new": message, "updated": updated}
        count += 1
        if count % 50 == 0:
            print(count)
    with open("test/preprocessed.json", "w") as f:
        json.dump(new, f)


async def test():
    testmsgs = [
        "take half my eth on arbitrum and swap it for usde on mainnet. then deposit it in the sy-usde pool on pendle",
    ]
    for t in testmsgs:
        print(
            await process_message_ai(
                t.lower(),
                user_address="",
                # save=4,
                skip_db=True,
                default_chain="ethereum",
            )
        )


async def runtest():
    with open("test/testdata.json", "r") as f:
        data = json.load(f)
    testmsgs = list(data.keys())
    for ix, t in enumerate(testmsgs):
        if ix % 2 == 0:
            continue
        print(
            await process_message_ai(
                t.lower(),
                user_address="",
                save=99,
                skip_db=True,
                default_chain="",
            )
        )
        print("")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("test suite")
        i = int(sys.argv[1])
        j = int(sys.argv[2])
        print("step: ", i)
        print("save: ", j)
        asyncio.run(suite(step=i, prompt=1, save=j))
    elif len(sys.argv) > 1:
        task = sys.argv[1]
        print(task)
        if task == "preprocess":
            asyncio.run(run_preprocess())
        elif task == "standardize":
            standardize()
        elif task == "split":
            split()
        elif task == "generate":
            asyncio.run(generate())
        elif task == "runtest":
            asyncio.run(runtest())
        else:
            print("what??")
    else:
        print("test")
        asyncio.run(test())

# TODO: conflict between camelot only on arbitrum and connected wallet base for subsequent calls
# add list of tokens to deposit action?
