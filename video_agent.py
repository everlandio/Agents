from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

@tool
def get_foreground_description(base64Frames):
    """Useful for identifying and descripting foreground objects in an sequence of frames."""
    llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1028)
    print("base64Frames", base64Frames)
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames from a video that I want you describe. Describe the foreground objects in the video and count the number of objects in each frame.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames),
            ],
        },
    ]
    response = llm.invoke(PROMPT_MESSAGES)
    return response

@tool
def get_background_description(base64Frames):
    """Useful for identifying and descripting background objects in an sequence of frames."""
    llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1028)
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames from a video that I want you describe. Describe the background objects in the video and when they become foreground objects.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames),
            ],
        },
    ]
    response = llm.invoke(PROMPT_MESSAGES)
    return response

import cv2
import base64

video = cv2.VideoCapture("./trem.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
base64Frames = base64Frames[0::40]
tools = [get_foreground_description, get_background_description]
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)

# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)

from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
import base64
class AgentState(TypedDict):
    input: str
    base64Frames: list[str]
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

from langgraph.prebuilt.tool_executor import ToolExecutor

tool_executor = ToolExecutor(tools)

def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}

def execute_tools(data):
    agent_action = data["agent_outcome"]
    print(agent_action)
    agent_action.tool_input = {"base64Frames": data["base64Frames"]}
    print(agent_action.tool_input)
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, output)]}

def should_continue(data):
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"

from langgraph.graph import END, StateGraph

workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")
app = workflow.compile()

inputs = {"input": "Describe all elements in this video (foreground and background)", "base64Frames": base64Frames, "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")