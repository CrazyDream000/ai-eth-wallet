import openai
import os
import requests
import re
from bs4 import BeautifulSoup
from openai.embeddings_utils import (
    get_embedding,
    get_embeddings,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
subscription_key = os.getenv("BING_SEARCH_V7_SUBSCRIPTION_KEY")

# does Arbitrum have stETH
# which liquid staking tokens (lido, fraxeth) have the highest yield?
# how is yield accrued for steth
# what is gas currently, what has the average gas been this week
query = "which liquid staking tokens (lido, fraxeth) have the highest yield?"


def fetch_chunks(url):
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
            },
        )
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            text = " ".join(soup.stripped_strings)
            return pd.DataFrame(
                {"url": url, "chunk": re.findall(r"(?:.|\n){1,987}(?:\. |$)", text)}
            )
        else:
            return None
    except Exception as e:
        print(f"Error fetching content: {e}")
        return None


def process_urls(urls):
    return pd.concat([fetch_chunks(url) for url in urls], ignore_index=True)


def embedding_search(query, df):
    embedding = get_embedding(query, engine="text-embedding-ada-002")
    df["embedding"] = get_embeddings(df["chunk"], engine="text-embedding-ada-002")
    df["similarity"] = distances_from_embeddings(
        embedding, df["embedding"], distance_metric="cosine"
    )
    idx = indices_of_nearest_neighbors_from_distances(df["similarity"]).tolist()
    return df.iloc[idx]


def bing_search(query):
    try:
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": subscription_key},
            params={"q": query, "mkt": "en-US"},
        )
        response.raise_for_status()
        return [res["url"] for res in response.json()["webPages"]["value"]]
    except Exception as ex:
        raise ex


def google_search(query):
    try:
        response = requests.get(
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
    except Exception as ex:
        raise ex


urls0 = google_search(query)
urls = [url for url in urls0 if re.search(r"beincrypto", url) == None]
df = process_urls(urls[:2])
res = embedding_search(query, df)
prompt = res["chunk"].head(3).tolist()

messages = [
    {
        "role": "system",
        "content": "give a concise answer, based on the context provided by the user.",
    },
    {"role": "user", "content": f"question: {query}\n\ncontext: {prompt}"},
]
response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages)
print(response)