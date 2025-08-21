# Import relevant functionality
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

#LLM = brain (thinks, generates text).
#LangChain = skeleton/framework that makes the brain act with tools, memory, agents.

# Create the agent
# MemorySaver is used to store the state of the agent
# this stores the prev conversation 
memory = MemorySaver()
# init_chat_model is used to create a chat model
## instead of anthropic:claude-3-5-sonnet-latest, we use gemini-2.5-flash
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="AIzaSyAl7GrG_qYTwKrExSdWZJniMlbNj_8LHfc")


# TavilySearch is used to search the web
search = TavilySearch(max_results=2, tavily_api_key="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") 
tools = [search]
# create_react_agent is used to create a react agent
# a ReAct Agent = an LLM that can reason, use tools, observe results, and refine its reasoning until it gives you the answer.
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}

input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in SF.",
}

# Instead of running all at once, the agent is streaming its reasoning process step by step.
#Each step contains updated conversation messages (human, AI thoughts, tool calls, tool outputs, etc.).
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

# here this will give the output with the recent result + prev saved result from memory = MemorySaver()
input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}

for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()