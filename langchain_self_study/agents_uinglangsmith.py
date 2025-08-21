# langsmith api key : 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

from langchain_tavily import TavilySearch

search = TavilySearch(max_results=2, tavily_api_key="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") 
# invoke is used to invoke the search
# search_results is the result of the search that is returned by the search tool
search_results = search.invoke("What is the weather in SF")
print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]