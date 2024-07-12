from cmath import inf
from langchain_core.tools import tool
import streamlit as st
import ui_template as ui
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from typing import List, Union
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text
import Tools
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain.chains import create_sql_query_chain

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool



def get_engine():
    engine = None
    if st.session_state.engine is None:
        df = Tools.load_csv_files(Tools.PATH, key='dataframe')
        st.session_state.df = df
        print("\n")
        print(df.head())
        print("\n")
        # setting columns as session state
        st.session_state.columns = list(df.columns)
        # print("\n\nColumns: ", st.session_state.columns)
        engine = create_engine("sqlite:///db.db")

        # delete the database if it exists
        connection = engine.connect()
        connection.execute(text('DROP TABLE IF EXISTS db'))
        # drop all tables in the database
        connection.execute(text('DROP TABLE IF EXISTS db'))

        df.to_sql("db", engine, index=False)
        
        st.session_state.engine = engine
    else:
        engine = st.session_state.engine
    return engine

def remove_markdown_code_block(sql_code):
    """
    Removes the Markdown code block formatting from a SQL code string.
    """
    # Check if the string starts with ```sql and ends with ```
    if sql_code.startswith("```sql") and sql_code.endswith("```"):
        # Remove the first 5 characters (```sql) and the last 3 characters (```)
        return sql_code[6:-3].strip()
    
    print("SQL Code: ", sql_code)
    return sql_code

@tool
def database_tool(query: str):
    """Use this to perform SELECT queries to get information from the database that has a table 'db' containing the user's uploaded data."""    
    db = SQLDatabase(engine=get_engine())
    execute_query = QuerySQLDataBaseTool(db=db,verbose=True,handle_tool_error=True)
    write_query = create_sql_query_chain(st.session_state.llm, db,k=inf)
    
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
    )

    chain = (
        RunnablePassthrough.assign(query=write_query|remove_markdown_code_block).assign(
            result=itemgetter("query") | execute_query
        )
        | answer_prompt
        | st.session_state.llm
        | StrOutputParser()
    )
    
    return chain.invoke({"question": query})

@tool
def describe_dataset(query: str):
    """Use this to describe the dataset, provide basic statistics, or answer questions about the structure of the data without performing SQL queries."""
    df = st.session_state.df
    
    if "describe" in query.lower() or "statistics" in query.lower():
        description = df.describe().to_string()
        return f"Here's a statistical description of the numerical columns in the dataset:\n{description}"
    
    elif "columns" in query.lower() or "fields" in query.lower():
        columns = ", ".join(df.columns)
        return f"The dataset contains the following columns: {columns}"
    
    elif "shape" in query.lower() or "size" in query.lower():
        rows, cols = df.shape
        return f"The dataset has {rows} rows and {cols} columns."
    
    elif "sample" in query.lower():
        sample = df.head().to_string()
        return f"Here's a sample of the first few rows of the dataset:\n{sample}"
    
    else:
        return "I'm sorry, I couldn't understand your request about the dataset. Could you please be more specific? You can ask about the dataset's description, statistics, columns, shape, or a sample of the data."


def get_tools():

    tools = [
        describe_dataset,
        database_tool,
    ]

    return tools

def get_memory():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory


def get_conversation_chain(agent, tools, memory):

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory = memory,
        )
    print("Agent Executor ready")
    return agent_executor



# each time user inputs a question, this function will be called
def handle_userinput(user_question):
    response = st.session_state.conversation.run(user_question)
    st.session_state.chat_history.append({"role": "human", "content": user_question})
    st.session_state.chat_history.append({"role": "ai", "content": response})

    for message in st.session_state.chat_history:
        if message["role"] == "human":
            st.write(ui.user_template.replace('{{MSG}}', message["content"]), unsafe_allow_html=True)
        else:
            st.write(ui.bot_template.replace('{{MSG}}', message["content"]), unsafe_allow_html=True)


def get_agent(tools):
    prefix = """You are an AI assistant that helps users analyze CSV data. 
    You have access to a database containing the user's uploaded CSV data in a table named 'db'.
    Use the database_tool to query this data and answer the user's questions.
    Always use SQL SELECT queries to retrieve information.

    If you don't know the answer or can't find the information in the database, simply say so.
    """
    suffix = """Begin!

    {chat_history}
    Human: {input}
    AI: Let me help you with that. I'll query the database to find the information you need.

    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=st.session_state.llm, prompt=prompt, verbose=True)
    
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    return agent


# this will be called first called from the Main.py
def initChat():
    st.write(ui.CSS, unsafe_allow_html=True)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'df' not in st.session_state:
        st.session_state.df = None

    engine = get_engine()
    tools = get_tools()
    memory = get_memory()
    agent = get_agent(tools)
    

    st.session_state.conversation = get_conversation_chain(agent, tools, memory)
    
