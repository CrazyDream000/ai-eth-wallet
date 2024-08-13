from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import os
import yaml
from langchain.llms.openai import OpenAI
from langchain import HuggingFaceHub
# from langchain.agents.agent_toolkits.openapi import planner
# from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
import json
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import yaml
from pydantic import Field

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.memory import ReadOnlySharedMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.tools.requests.tool import BaseRequestsTool
from langchain.utilities.requests import RequestsWrapper

# flake8: noqa

from langchain.prompts.prompt import PromptTemplate


API_PLANNER_PROMPT = """You are a planner that plans a sequence of API calls to assist with user queries against an API.

You should:
1) evaluate whether the user query can be solved by the API documentated below. If no, say why.
2) if yes, generate a plan of API calls and say what they are doing step by step.

You should only use API endpoints documented below ("Endpoints you can use:").
Some user queries can be resolved in a single API call, but some will require several API calls.
The plan will be passed to an API controller that can format it into web requests and return the responses.

----

Here are some examples:

Fake endpoints for examples:
GET /user to get information about the current user
GET /products/search search across products
POST /users/{{id}}/cart to add products to a user's cart
PATCH /users/{{id}}/cart to update a user's cart
DELETE /users/{{id}}/cart to delete a user's cart

User query: tell me a joke
Plan: Sorry, this API's domain is shopping, not comedy.

User query: I want to buy a couch
Plan: 1. GET /products with a query param to search for couches
2. GET /user to find the user's id
3. POST /users/{{id}}/cart to add a couch to the user's cart

User query: I want to add a lamp to my cart
Plan: 1. GET /products with a query param to search for lamps
2. GET /user to find the user's id
3. PATCH /users/{{id}}/cart to add a lamp to the user's cart

User query: I want to delete my cart
Plan: 1. GET /user to find the user's id
2. DELETE required. Did user specify DELETE or previously authorize? Yes, proceed.
3. DELETE /users/{{id}}/cart to delete the user's cart

User query: I want to start a new cart
Plan: 1. GET /user to find the user's id
2. DELETE required. Did user specify DELETE or previously authorize? No, ask for authorization.
3. Are you sure you want to delete your cart? 
----

Here are endpoints you can use. Do not reference any of the endpoints above.

{endpoints}

----

The endpoints you can use are all related to performing blockchain transactions. 
The user query below tell you which endpoints to call and you need to properly call the endpoints in the correct order.

User query: {query}
Plan:"""
API_PLANNER_TOOL_NAME = "api_planner"
API_PLANNER_TOOL_DESCRIPTION = f"Can be used to generate the right API calls to assist with a user query, like {API_PLANNER_TOOL_NAME}(query). Should always be called before trying to call the API controller."

# Execution.
API_CONTROLLER_PROMPT = """You are an agent that gets a sequence of API calls and given their documentation, should execute them and return the final response.
If you cannot complete them and run into issues, you should explain the issue. If you're unable to resolve an API call, you probably chose the wrong inputs, and should reconsider and rebuild the Action Input. 


Here is documentation on the API:
Base url: {api_url}
Endpoints:
{api_docs}


Here are tools to execute requests against the API: {tool_descriptions}


Starting below, you should follow this format:

Plan: the plan of API calls to execute
Thought: you should always think about what to do
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the output of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing the plan (or, I cannot finish executing the plan without knowing some other information.)
Final Answer: the final output from executing the plan or missing information I'd need to re-plan correctly.

The output of the action will be a transaction object. 
The blockchain address stored in the to field is a smart contract that is being called, and what is being passed to this smart contract is the string stored in the data field.


Begin!

Plan: {input}
Thought:
{agent_scratchpad}
"""
API_CONTROLLER_TOOL_NAME = "api_controller"
API_CONTROLLER_TOOL_DESCRIPTION = f"Can be used to execute a plan of API calls, like {API_CONTROLLER_TOOL_NAME}(plan)."

# Orchestrate planning + execution.
# The goal is to have an agent at the top-level (e.g. so it can recover from errors and re-plan) while
# keeping planning (and specifically the planning prompt) simple.
API_ORCHESTRATOR_PROMPT = """You are an agent that assists with user queries against API, things like querying information or creating resources.
Some user queries can be resolved in a single API call, particularly if you can find appropriate params from the OpenAPI spec; though some require several API calls.
You should always plan your API calls first, and then execute the plan second.
You should never return information without executing the api_controller tool.
You should always make sure the entire user query is captured in the first Action Input. You should not be missing any data from the user query.
Some user queries will list the steps you should take. Some user queries will be more abstract and require knowledge of how blockchain transactions work.


Here are the tools to plan and execute API requests: {tool_descriptions}


Starting below, you should follow this format:

User query: the query a User wants help with related to the API
Thought: you should always think about what to do
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for or the data the user asked to create
Final Answer: the final output from executing the plan


Example:
User query: can you add some trendy stuff to my shopping cart.
Thought: I should plan API calls first.
Action: api_planner
Action Input: I need to find the right API calls to add trendy items to the users shopping cart
Observation: 1) GET /items with params 'trending' is 'True' to get trending item ids
2) GET /user to get user
3) POST /cart to post the trending items to the user's cart
Thought: I'm ready to execute the API calls.
Action: api_controller
Action Input: 1) GET /items params 'trending' is 'True' to get trending item ids
2) GET /user to get user
3) POST /cart to post the trending items to the user's cart
...

Begin!

User query: {input}
Thought: I should generate a plan to help with this query and then copy that plan exactly to the controller.
{agent_scratchpad}"""


REQUESTS_GET_TOOL_DESCRIPTION = """Use this to GET content from a website.
Input to the tool should be a json string with 2 keys: "url", "params".
The value of "url" should be a string. 
The value of "params" should be a dict of the needed and available parameters from the OpenAPI spec related to the endpoint. 
If parameters are not needed, or not available, leave it empty.
"""

PARSING_GET_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract information from the API response.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response"],
)

REQUESTS_POST_TOOL_DESCRIPTION = """Use this when you want to POST to a website.
Input to the tool should be a json string with 2 keys: "url", "data".
The value of "url" should be a string.
The value of "data" should be a dictionary of key-value pairs you want to POST to the url.
Always use double quotes for strings in the json string."""


PARSING_POST_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
The response is a blockchain transaction object.
Your task is to return the transaction object.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response"],
)


# @tool
# def get_word_length(word: str) -> int:
    # """Returns the length of a word."""
    # return len(word)
# 
# tools = [get_word_length]
# llm = ChatOpenAI(temperature=0, openai_api_key='')
# system_message = SystemMessage(content="You are very powerful assistant, but bad at calculating lengths of words.")
# # prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
# MEMORY_KEY = "chat_history"
# prompt = OpenAIFunctionsAgent.create_prompt(
    # system_message=system_message,
    # extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
# )
# memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
# agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
# agent_executor.run("how many letters in the word educa?")
# agent_executor.run("is that a real word?")

MAX_RESPONSE_LENGTH = 5000
"""Maximum length of the response to be returned."""


def _get_default_llm_chain(prompt: BasePromptTemplate) -> LLMChain:
    return LLMChain(
        llm=ChatOpenAI(),
        prompt=prompt,
    )


def _get_default_llm_chain_factory(
    prompt: BasePromptTemplate,
) -> Callable[[], LLMChain]:
    """Returns a default LLMChain factory."""
    return partial(_get_default_llm_chain, prompt)

class RequestsGetToolWithParsing(BaseRequestsTool, BaseTool):
    """Requests GET tool with LLM-instructed extraction of truncated responses."""

    name = "requests_get"
    """Tool name."""
    description = REQUESTS_GET_TOOL_DESCRIPTION
    """Tool description."""
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_GET_PROMPT)
    )
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        data_params = data.get("params")
        response = self.requests_wrapper.get(data["url"], params=data_params)
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()
    
class RequestsPostToolWithParsing(BaseRequestsTool, BaseTool):
    """Requests POST tool with LLM-instructed extraction of truncated responses."""

    name = "requests_post"
    """Tool name."""
    description = REQUESTS_POST_TOOL_DESCRIPTION
    """Tool description."""
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_POST_PROMPT)
    )
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            if text[:3] == "```":
                try:
                    data = json.loads(text[7:-4])
                except json.JSONDecodeError as e:
                    raise e
            raise e
        response = self.requests_wrapper.post(data["url"], data["data"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()
    
#
# Orchestrator, planner, controller.
#
def _create_api_planner_tool(
    api_spec: ReducedOpenAPISpec, llm: BaseLanguageModel
) -> Tool:
    endpoint_descriptions = [
        f"{name} {description}" for name, description, _ in api_spec.endpoints
    ]
    prompt = PromptTemplate(
        template=API_PLANNER_PROMPT,
        input_variables=["query"],
        partial_variables={"endpoints": "- " + "- ".join(endpoint_descriptions)},
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tool = Tool(
        name=API_PLANNER_TOOL_NAME,
        description=API_PLANNER_TOOL_DESCRIPTION,
        func=chain.run,
    )
    return tool

def _create_api_controller_agent(
    api_url: str,
    api_docs: str,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
) -> AgentExecutor:
    get_llm_chain = LLMChain(llm=llm, prompt=PARSING_GET_PROMPT)
    post_llm_chain = LLMChain(llm=llm, prompt=PARSING_POST_PROMPT)
    tools: List[BaseTool] = [
        RequestsGetToolWithParsing(
            requests_wrapper=requests_wrapper, llm_chain=get_llm_chain
        ),
        RequestsPostToolWithParsing(
            requests_wrapper=requests_wrapper, llm_chain=post_llm_chain
        ),
    ]
    prompt = PromptTemplate(
        template=API_CONTROLLER_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "api_url": api_url,
            "api_docs": api_docs,
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        allowed_tools=[tool.name for tool in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

def _create_api_controller_tool(
    api_spec: ReducedOpenAPISpec,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
) -> Tool:
    """Expose controller as a tool.

    The tool is invoked with a plan from the planner, and dynamically
    creates a controller agent with relevant documentation only to
    constrain the context.
    """

    base_url = api_spec.servers[0]["url"]  # TODO: do better.

    def _create_and_run_api_controller_agent(plan_str: str) -> str:
        pattern = r"\b(GET|POST|PATCH|DELETE)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [
            "{method} {route}".format(method=method, route=route.split("?")[0])
            for method, route in matches
        ]
        endpoint_docs_by_name = {name: docs for name, _, docs in api_spec.endpoints}
        docs_str = ""
        for endpoint_name in endpoint_names:
            docs = endpoint_docs_by_name.get(endpoint_name)
            if not docs:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")
            docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"

        agent = _create_api_controller_agent(base_url, docs_str, requests_wrapper, llm)
        return agent.run(plan_str)

    return Tool(
        name=API_CONTROLLER_TOOL_NAME,
        func=_create_and_run_api_controller_agent,
        description=API_CONTROLLER_TOOL_DESCRIPTION,
    )

def create_openapi_agent(
    api_spec: ReducedOpenAPISpec,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
    shared_memory: Optional[ReadOnlySharedMemory] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    verbose: bool = True,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Instantiate OpenAI API planner and controller for a given spec.

    Inject credentials via requests_wrapper.

    We use a top-level "orchestrator" agent to invoke the planner and controller,
    rather than a top-level planner
    that invokes a controller with its plan. This is to keep the planner simple.
    """
    tools = [
        _create_api_planner_tool(api_spec, llm),
        _create_api_controller_tool(api_spec, requests_wrapper, llm),
    ]
    prompt = PromptTemplate(
        template=API_ORCHESTRATOR_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt, memory=shared_memory),
        allowed_tools=[tool.name for tool in tools],
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )

"""Quick and dirty representation for OpenAPI specs."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union


def dereference_refs(spec_obj: dict, full_spec: dict) -> Union[dict, list]:
    """Try to substitute $refs.

    The goal is to get the complete docs for each endpoint in context for now.

    In the few OpenAPI specs I studied, $refs referenced models
    (or in OpenAPI terms, components) and could be nested. This code most
    likely misses lots of cases.
    """

    def _retrieve_ref_path(path: str, full_spec: dict) -> dict:
        components = path.split("/")
        if components[0] != "#":
            raise RuntimeError(
                "All $refs I've seen so far are uri fragments (start with hash)."
            )
        out = full_spec
        for component in components[1:]:
            out = out[component]
        # print(out)
        return out

    def _dereference_refs(
        obj: Union[dict, list], stop: bool = False
    ) -> Union[dict, list]:
        if stop:
            return obj
        obj_out: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "$ref":
                    # stop=True => don't dereference recursively.
                    return _dereference_refs(
                        _retrieve_ref_path(v, full_spec), stop=True
                    )
                elif isinstance(v, list):
                    obj_out[k] = [_dereference_refs(el) for el in v]
                elif isinstance(v, dict):
                    obj_out[k] = _dereference_refs(v)
                else:
                    obj_out[k] = v
            return obj_out
        elif isinstance(obj, list):
            return [_dereference_refs(el) for el in obj]
        else:
            return obj

    return _dereference_refs(spec_obj)



@dataclass(frozen=True)
class ReducedOpenAPISpec:
    servers: List[dict]
    description: str
    endpoints: List[Tuple[str, str, dict]]


def reduce_openapi_spec(spec: dict, dereference: bool = True) -> ReducedOpenAPISpec:
    """Simplify/distill/minify a spec somehow.

    I want a smaller target for retrieval and (more importantly)
    I want smaller results from retrieval.
    I was hoping https://openapi.tools/ would have some useful bits
    to this end, but doesn't seem so.
    """
    # 1. Consider only get, post, patch, delete endpoints.
    endpoints = [
        (f"{operation_name.upper()} {route}", docs.get("description"), docs)
        for route, operation in spec["paths"].items()
        for operation_name, docs in operation.items()
        if operation_name in ["get", "post", "patch", "delete"]
    ]

    # 2. Replace any refs so that complete docs are retrieved.
    # Note: probably want to do this post-retrieval, it blows up the size of the spec.
    if dereference:
        endpoints = [
            (name, description, dereference_refs(docs, spec))
            for name, description, docs in endpoints
        ]
    # print(endpoints)

    # 3. Strip docs down to required request args + happy path response.
    def reduce_endpoint_docs(docs: dict) -> dict:
        out = {}
        if docs.get("description"):
            out["description"] = docs.get("description")
        if docs.get("parameters"):
            out["parameters"] = [
                parameter
                for parameter in docs.get("parameters", [])
                if parameter.get("required")
            ]
        if "200" in docs["responses"]:
            out["responses"] = docs["responses"]["200"]
        return out

    # endpoints = [
        # (name, description, reduce_endpoint_docs(docs))
        # for name, description, docs in endpoints
    # ]
    # print(endpoints)

    return ReducedOpenAPISpec(
        servers=spec["servers"],
        description=spec["info"].get("description", ""),
        endpoints=endpoints,
    )

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

with open("wallet.yaml") as f:
    raw_wallet_api_spec = yaml.load(f, Loader=yaml.Loader)
wallet_api_spec = reduce_openapi_spec(raw_wallet_api_spec)
# print(wallet_api_spec)
# llm = HuggingFaceHub(repo_id='augtoma/qCammel-70-x', model_kwargs={'temperature':1e-10})
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.01, openai_api_key='')
wallet_agent = create_openapi_agent(wallet_api_spec, {}, llm) #, agent_executor_kwargs={'max_iterations': 2})
# user_query = ("Swap 1 ETH for USDC on Ethereum")
# print(wallet_agent.__call__(user_query, include_run_info=True))
# user_query = ("Bridge 1 ETH from Ethereum to Arbitrum")
# print(wallet_agent.__call__(user_query, include_run_info=True))

user_query = ("Swap 1 ETH for USDC on Ethereum then bridge to Arbitrum")
# user_query = ("I have 1 ETH on Ethereum and I want ARB on Arbitrum")
print(wallet_agent.__call__(user_query, include_run_info=True))