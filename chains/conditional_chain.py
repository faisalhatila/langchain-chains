from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation"
)

# model1 = ChatOpenAI()
model = ChatHuggingFace(llm=llm)
# model = ChatOpenAI()
parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction} ',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# prompt1 = PromptTemplate(
#     template='Classify the sentiment of the following feedback text into positive or negative \n {feedback}',
#     input_variables=['feedback']
# )


classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# result = classifier_chain.invoke({'feedback':'This is an amazing smartphone'}).sentiment

# print(result)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive',prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative',prompt3 | model | parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'This is Faisal I purchased the smart phone I must say I just throwed my money into gutter. Totally worst smartphone I ever had camera, battery all crap'})

print(result)

chain.get_graph().print_ascii()