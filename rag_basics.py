from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Create embeddings
embeddings = OpenAIEmbeddings()

# Step 2: Create a vectorstore from some sample texts
sample_texts = [
    "An autonomous agent system has a memory module.",
    "It also has a planning module.",
    "An execution component carries out the planned actions.",
    "LLM-powered agents often use a retriever to fetch relevant knowledge."
]
vectorstore = FAISS.from_texts(sample_texts, embeddings)

# Step 3: Turn vectorstore into a retriever
retriever = vectorstore.as_retriever()

# Define a system prompt that tells the model how to use the retrieved context
system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Context: {context}:"""

# Define a question
question = """What are the main components of an LLM-powered autonomous agent system?"""

# Retrieve relevant documents
docs = retriever.invoke(question)

# Combine the documents into a single string
docs_text = "".join(d.page_content for d in docs)

# Populate the system prompt with the retrieved context
system_prompt_fmt = system_prompt.format(context=docs_text)

# Create a model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Generate a response
questions = model.invoke([SystemMessage(content=system_prompt_fmt),
                          HumanMessage(content=question)])
