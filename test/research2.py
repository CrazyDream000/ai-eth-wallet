import openai
import os
import requests
import re
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding, get_embeddings, distances_from_embeddings, indices_of_nearest_neighbors_from_distances
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")
subscription_key = os.getenv("BING_SEARCH_V7_SUBSCRIPTION_KEY")

# does Arbitrum have stETH
# which liquid staking tokens (lido, fraxeth) have the highest yield?
# how is yield accrued for steth
# what is gas currently, what has the average gas been this week
query = "how is yield accrued for steth"


def fetch_snippets(url):
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0"
            },
        )
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            text = " ".join(soup.stripped_strings)
            return pd.DataFrame(
                {"url": url, "snippet": re.findall(r"(?:.|\n){1,987}(?:\. |$)", text)}
            )
        else:
            return None
    except Exception as e:
        print(f"Error fetching content: {e}")
        return None


def process_results(results, pages):
    return pd.concat(
        [fetch_snippets(res["url"]) for res in results[:pages]], ignore_index=True
    )


def search(query, df):
    embedding = get_embedding(query, engine="text-embedding-ada-002")
    df["embedding"] = get_embeddings(df["snippet"], engine="text-embedding-ada-002")
    df["similarity"] = distances_from_embeddings(embedding, df['embedding'], distance_metric="cosine")
    idx = indices_of_nearest_neighbors_from_distances(df['similarity']).tolist()
    return df.iloc[idx]


try:
    response = requests.get(
        "https://api.bing.microsoft.com/v7.0/search",
        headers={"Ocp-Apim-Subscription-Key": subscription_key},
        params={"q": query, "mkt": "en-US"},
    )
    response.raise_for_status()
except Exception as ex:
    raise ex

df = process_results(response.json()["webPages"]["value"], pages=5)
res = search(query, df)
prompt = res["snippet"].head(5).tolist()

messages = [
    {
        "role": "system",
        "content": "give a concise answer, based on the context provided by the user",
    },
    {"role": "user", "content": f"question: {query}\n\ncontext: {prompt}"},
]
response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages)
print(response)