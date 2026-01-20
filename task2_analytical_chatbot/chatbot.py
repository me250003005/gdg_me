from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import streamlit as st
from tools import fetch_news, analyze_stock

# LLM Setup
llm = HuggingFacePipeline(pipeline=pipeline("text-generation", model="gpt2"))

# Agent Prompt
prompt = PromptTemplate.from_template(
    "You are a financial advisor. Use tools to explain stock movements. Answer queries like 'Why did AAPL drop?' by fetching news and analyzing data."
)

# Agent
tools = [fetch_news, analyze_stock]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

def main():
    st.title("Capital Pulse - Analytical Chatbot")
    query = st.text_input("Ask about stocks (e.g., 'Why did AAPL drop?')")
    if query:
        response = executor.invoke({"input": query})
        st.write(response['output'])

if __name__ == "__main__":
    main()
