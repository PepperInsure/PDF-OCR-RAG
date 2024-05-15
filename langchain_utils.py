from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def setup_chain():
    review_template_str = """Your task is to use the extracted content from OCR data to answer questions about a document.
Use the following context to answer the questions. Be as detailed as possible, but do not make up any information
that is not from the context. If you don't know an answer, say you don't know. Give one answer in english and one in japanese.
{context}
"""
    review_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"], template=review_template_str
        )
    )
    review_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["question"], template="{question}")
    )
    messages = [review_system_prompt, review_human_prompt]
    review_prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"], messages=messages
    )
    # use the new gpt4-o said to be better for multiple languages
    chat_model = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    output_parser = StrOutputParser()

    return (
            review_prompt_template
            | chat_model
            | output_parser
    )