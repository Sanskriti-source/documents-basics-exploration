import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
# uses beautifulsoup to parse the html and extract the text (bs4)
loader = WebBaseLoader(
    # this fetches the contents from github repo of the blog
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # post-content is the main content of the blog
            # post-title is the title of the blog
            # post-header is the header of the blog
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
# bs_kwargs = beautifulsoup keyowrd arg
# bs_kwargs -> controls what is to be extracted from the html
docs = loader.load()

# text_splitter - breaks the executed text into smaller chuncks as it is easy to get it executed 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# split_documents - splits the documents into smaller chunks that will go to vector store for retrieval later
all_splits = text_splitter.split_documents(docs)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

#  LLM 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyAl7GrG_qYTwKrExSdWZJniMlbNj_8LHfc")

# here FAISS assigns the index numbers from 0 to infinity to each chuncks inside vecdb 
vector_store = FAISS.from_documents(docs, embeddings)


# here the documents are added to the vectore store which contains combined chuncks like earlier & their corresponding vec numbers along with indexes assigned to each chunck using FAISS
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
# here i don't have to put the API key but just it will fetch from itself 
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# RETRIVAL
# Define application steps
def retrieve(state: State):
    # it retrieves the content having similarity score approx equal to the user's query 
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

from langchain_google_genai import ChatGoogleGenerativeAI
llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="AIzaSyAl7GrG_qYTwKrExSdWZJniMlbNj_8LHfc")

# AUGMENTATION
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# GENERATION
# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

inputs = {"question": "What is an agent in AI as per your knowledge?"}
result = graph.invoke(inputs)
print(result)
