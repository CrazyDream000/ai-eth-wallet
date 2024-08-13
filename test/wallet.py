import os
import yaml
import openai
from langchain.llms.openai import OpenAI
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
# from langchain.agents.agent_toolkits.openapi import planner
# from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
openai.api_key = ""
"""Agent that interacts with OpenAPI APIs via a hierarchical planning approach."""
import json
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import yaml
from pydantic import Field

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.openapi.planner_prompt import (
    API_CONTROLLER_TOOL_DESCRIPTION,
    API_CONTROLLER_TOOL_NAME,
    API_ORCHESTRATOR_PROMPT,
    API_PLANNER_TOOL_DESCRIPTION,
    API_PLANNER_TOOL_NAME,
    API_PLANNER_PROMPT,
    PARSING_DELETE_PROMPT,
    PARSING_GET_PROMPT,
    PARSING_PATCH_PROMPT,
    PARSING_POST_PROMPT,
    REQUESTS_DELETE_TOOL_DESCRIPTION,
    REQUESTS_GET_TOOL_DESCRIPTION,
    REQUESTS_PATCH_TOOL_DESCRIPTION,
    REQUESTS_POST_TOOL_DESCRIPTION,
)
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


# Execution.
API_CONTROLLER_PROMPT = """You are an agent that gets a sequence of API calls and given their documentation, should execute them and return the final response.
If you cannot complete them and run into issues, you should explain the issue. If you're unable to resolve an API call, you can retry the API call. Perform API calls until the entire User query is satisfied. 
Return results from the API call directly to the User, in JSON format.


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


All Observations should be JSON format. Begin!

Plan: {input}
Thought:
{agent_scratchpad}
"""
#
# Requests tools with LLM-instructed extraction of truncated responses.
#
# Of course, truncating so bluntly may lose a lot of valuable
# information in the response.
# However, the goal for now is to have only a single inference step.
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
            response=response, instructions=data["output_instructions"]
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
            raise e
        response = self.requests_wrapper.post(data["url"], data["data"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()



class RequestsPatchToolWithParsing(BaseRequestsTool, BaseTool):
    """Requests PATCH tool with LLM-instructed extraction of truncated responses."""

    name = "requests_patch"
    """Tool name."""
    description = REQUESTS_PATCH_TOOL_DESCRIPTION
    """Tool description."""
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_PATCH_PROMPT)
    )
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.patch(data["url"], data["data"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()



class RequestsDeleteToolWithParsing(BaseRequestsTool, BaseTool):
    """A tool that sends a DELETE request and parses the response."""

    name = "requests_delete"
    """The name of the tool."""
    description = REQUESTS_DELETE_TOOL_DESCRIPTION
    """The description of the tool."""

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    """The maximum length of the response."""
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_DELETE_PROMPT)
    )
    """The LLM chain used to parse the response."""

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.delete(data["url"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
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
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0.01)
wallet_agent = create_openapi_agent(wallet_api_spec, {}, llm) #, agent_executor_kwargs={'max_iterations': 2})
# user_query = ("Swap 1 ETH for USDC on Ethereum")
# print(wallet_agent.__call__(user_query, include_run_info=True))
# user_query = ("Bridge 1 ETH from Ethereum to Arbitrum")
# print(wallet_agent.__call__(user_query, include_run_info=True))

user_query = ("Swap 1 ETH for USDC on Ethereum then bridge to Arbitrum")
# user_query = ("I have 1 ETH on Ethereum and I want ARB on Arbitrum")
print(wallet_agent.__call__(user_query, include_run_info=True))