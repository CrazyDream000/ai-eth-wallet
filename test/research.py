import openai
import os, json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
subscription_key = os.getenv('BING_SEARCH_V7_SUBSCRIPTION_KEY')

class SearchResult:
    def __init__(self, title, link):
        self.title = title
        self.link = link
        self.summary = None
        self.full_content = None

    def to_dict(self):
        return { 'title': self.title, 'link': self.link, 'summary': self.summary, 'full_content': self.full_content }

def fetch_content(url, summary=False):
    """
    Fetches the content of the given URL.
    Returns a summary if the summary parameter is set to True.
    """
    try:
        response = requests.get(url, headers={'User-Agent': "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0"})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            text = ' '.join(soup.stripped_strings)
            return text[:900] + '...' if summary else text[:9000]
        else:
            print(url)
            print(response)
            return None
    except Exception as e:
        print(f"Error fetching content: {e}")
        return None

def process_results(results):
    formatted_results = [SearchResult(res['name'], res['url']) for res in results]

    for result in formatted_results[:5]:
        result.summary = fetch_content(result.link, summary=True) or "Error fetching summary"

    for result in formatted_results[:1]:
        result.full_content = fetch_content(result.link, summary=False) or "Error fetching content"

    return [res.to_dict() for res in formatted_results][:5]

def perform_research(query):
    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers={ 'Ocp-Apim-Subscription-Key': subscription_key }, params={ 'q': query, 'mkt': 'en-US' })
        response.raise_for_status()
        # print(json.dumps(response.json()))
    except Exception as ex:
        raise ex
    snippets = process_results(response.json()['webPages']['value'])
    # print(snippets)
    messages = [{"role": "system", "content": "give a concise answer, based on the context provided by the user. do not mention the word 'context' in your answer."},
                {"role": "user", "content": f'question: {query}\n\ncontext: {snippets}'}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages
    )
    print(response)
    try:
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return ""

perform_research("what is eth gas currently, what has the average eth gas been this week")