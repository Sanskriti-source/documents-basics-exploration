
# langsmith api key : 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

from langchain_tavily import TavilySearch

search = TavilySearch(max_results=2, tavily_api_key="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") 
# invoke is used to invoke the search
# search_results is the result of the search that is returned by the search tool
search_results = search.invoke("What is the weather in SF")
# print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

# MAIN CODE OF LANUGAGE MODEL STARTS FROM HERE 
## instead of anthropic:claude-3-5-sonnet-latest, we use gemini-2.5-flash
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="AIzaSyAl7GrG_qYTwKrExSdWZJniMlbNj_8LHfc")


query = "Hi!"
# here the model is invoked(triggered) with the query
# the response is the response from the model
# here the content = query = Hi! and role = user means user is asking the model to say hi
response = model.invoke([{"role": "user", "content": query}])
# the response.text() is the text of the response
response.text()
# bind_tools is used to bind the tools to the model
# tools is the list of tools that we want to bind to the model
# model_with_tools is the model with the tools bound to it
model_with_tools = model.bind_tools(tools)

# here there are no tools needed hence tool_calls is an empty list coz we r just calling Hi!
query = "Hi!"
response = model_with_tools.invoke([{"role": "user", "content": query}])

print(f"Message content: {response.text()}\n")
print(f"Tool calls: {response.tool_calls}")

# here tools which are needed are triggered / invoked and displayed hence not an empty list

query = "Search for the weather in SF"
response = model_with_tools.invoke([{"role": "user", "content": query}])

print(f"Message content: {response.text()}\n")
print(f"Tool calls: {response.tool_calls}")