from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain.agents import AgentExecutor, ZeroShotAgent
import streamlit as st
import src.chat_with_csv.ui_template as ui
from src.chat_with_csv.agent_tools import get_sqlite_engine, describe_dataset, database_tool,pretty_print_result, handle_unexpected_query


def get_tools():
    """
    Returns a list of tools.

    Returns:
        list: A list of tools.
    """
    tools = [
        describe_dataset,
        database_tool,
        pretty_print_result,
        handle_unexpected_query
    ]
    return tools


def get_memory():
    """
    Retrieves the conversation buffer memory.

    Returns:
        ConversationBufferMemory: The conversation buffer memory object.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory


def get_agent(tools):
    """
    Creates and returns an AI agent that helps users analyze CSV data.

    Args:
        tools (list): A list of tools available to the agent.

    Returns:
        agent (ZeroShotAgent): The AI agent that can analyze CSV data.

    """
    prefix = """You are an AI assistant that helps users analyze CSV data. 
    You have access to a database containing the user's uploaded CSV data in a table named 'db'.
    Use the database_tool to query this data and answer the user's questions.
    Always use SQL SELECT queries to retrieve information.
    
    IMPORTANT: When you receive results from the database_tool, use that information to answer the user's question.
    Do not claim you don't have access to the information if the database_tool returns results.

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

    llm_chain = LLMChain(llm=st.session_state.strict_llm, prompt=prompt, verbose=True)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    return agent


def get_conversation_chain(agent, tools, memory):
    """
    Retrieves the conversation chain for the given agent, tools, and memory.

    Args:
        agent: The agent object representing the conversational agent.
        tools: The tools object containing the necessary tools for the agent.
        memory: The memory object representing the agent's memory.

    Returns:
        An instance of AgentExecutor initialized with the provided agent, tools, and memory.
    """
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
    print("Agent Executor ready")
    return agent_executor


def initChat():
    """
    Initializes the chat application by setting up the necessary session state variables and objects.

    This function performs the following tasks:
    1. Sets up the CSS for the chat UI.
    2. Checks if the 'conversation' session state variable exists and initializes it if not.
    3. Checks if the 'chat_history' session state variable exists and initializes it if not.
    4. Checks if the 'engine' session state variable exists and initializes it if not.
    5. Checks if the 'df' session state variable exists and initializes it if not.
    6. Retrieves the SQLite engine.
    7. Retrieves the tools.
    8. Retrieves the memory.
    9. Retrieves the agent.
    10. Sets the 'conversation' session state variable to the conversation chain.

    Returns:
        None
    """
    st.write(ui.CSS, unsafe_allow_html=True)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'columns' not in st.session_state:
        st.session_state.columns = []
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'strict_llm' not in st.session_state:
        st.session_state.strict_llm = None

    get_sqlite_engine()
    tools = get_tools()
    memory = get_memory()
    agent = get_agent(tools)
    
    st.session_state.conversation = get_conversation_chain(agent, tools, memory)
    

def handle_userinput(user_question):
    response = st.session_state.conversation.run(user_question)
    st.session_state.chat_history.append({"role": "human", "content": user_question})
    st.session_state.chat_history.append({"role": "ai", "content": response})

    for message in st.session_state.chat_history:
        if message["role"] == "human":
            st.write(ui.user_template(message["content"]), unsafe_allow_html=True)
        else:
            st.write(ui.bot_template(message["content"]), unsafe_allow_html=True)
