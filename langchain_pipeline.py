from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Initialize the LLM
llm = OpenAI(temperature=0.7) 
# Replace with your preferred LLM and API key setup

# 2. Define a Prompt Template
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}"
)

# 3. Create an LLMChain
qa_chain = LLMChain(llm=llm, prompt=prompt)

# 4. Invoke the chain
question = "What is the capital of France?"
response = qa_chain.run(question)
print(response)