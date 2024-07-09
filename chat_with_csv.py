from langchain_core.tools import tool
import streamlit as st
import ui_template as ui
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
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
    """This is used to perform only SELECT query to get information from database that has table 'db' which contains data uploaded by the user."""    
    db = SQLDatabase(engine=get_engine())
    execute_query = QuerySQLDataBaseTool(db=db,verbose=True)
    write_query = create_sql_query_chain(st.session_state.llm, db)
    
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

def get_tools():

    tools = [
        database_tool,
    ]

    return tools

def get_memory():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory

def get_agent(tools, memory):

    # Set up the base template
    template = """Complete the objective as best and detailed as you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    These were previous tasks you completed:



    Begin!
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""


    # Set up a prompt template
    class CustomPromptTemplate(BaseChatPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]

        def format_messages(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            formatted = self.template.format(**kwargs)
            return [HumanMessage(content=formatted)]




    prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "chat_history" ,"intermediate_steps"]
    )

    class CustomOutputParser(AgentOutputParser):

        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                llm_output = 'Unable to answer the question. Please try again.'
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    output_parser = CustomOutputParser()
    llm = st.session_state.llm
    if(llm is None):
        print("LLM not ready")
        return None

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tools
    )
    print("Tools Ready")
    return agent




def get_conversation_chain(agent, tools, memory):

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory = memory
        )
    print("Agent Executor ready")
    return agent_executor


def handle_userinput(user_question):
    # response = st.session_state.conversation(user_question)
    response = st.session_state.conversation.invoke(user_question)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(ui.user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
        else:
            st.write(ui.bot_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)


def initChat():
    st.write(ui.CSS, unsafe_allow_html=True)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    get_engine()
    tools = get_tools()
    memory = get_memory()
    agent = get_agent(tools, memory)
    st.session_state.conversation = get_conversation_chain(agent, tools, memory)

    # from langchain_community.agent_toolkits import create_sql_agent

    # llm = st.session_state.llm
    # agent_executor = create_sql_agent(llm, db=SQLDatabase( st.session_state.engine), agent_type="openai-tools", verbose=True)
    # st.session_state.conversation = agent_executor
    
