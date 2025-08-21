
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain

# 1. Initialize Gemini with API key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key="AIzaSyAl7GrG_qYTwKrExSdWZJniMlbNj_8LHfc"  
)

# 2. Define prompt template
#prompt = PromptTemplate(
#    input_variables=["question"],
#    template="Answer the following question: {question}"
#)

# 2. Define prompt
prompt = PromptTemplate.from_template("Answer the following question: {question}")


# 3. Create chain
# LLMChain is a chain that combines a prompt template and a language model(llm created above).
# a basic LLMChain, which is the simplest chain type: Prompt → LLM → Output.
#qa_chain = LLMChain(llm=llm, prompt=prompt)

# Instead above create chain using Runnable syntax
chain = prompt | llm
# 4. Run
question = "What is the capital of France?"
response = chain.invoke({"question": question})

# Print only the text output
print("Answer:", response.content)

# 4. Run
#question = "What is the capital of France?"
#response = qa_chain.run(question)
#print(response)
#response = chain.invoke({"question": "What is the capital of France?"})
#print(response.content)

# 4. Run
#question = "What is the capital of France?"
#response = chain.invoke(question)
#print(response)


