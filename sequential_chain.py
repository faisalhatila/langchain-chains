from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)
model1 = ChatOpenAI()


parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
chain1 = prompt1 | model1 | parser | prompt2 | model1 | parser

result = chain.invoke({'topic':'Unemployment in Pakistan'})
result1 = chain1.invoke({'topic':'Unemployment in Pakistan'})

print(result)
print('**************************************************')
print(result1)
print('**************************************************')
chain.get_graph().print_ascii()
print('**************************************************')
chain1.get_graph().print_ascii()