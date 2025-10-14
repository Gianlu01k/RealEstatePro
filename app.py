import streamlit as st
import pandas as pd
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from agent import CustomAgentExecutor
from tools import retrieve, average_price, final_answer, calculate_travel_time, find_places_near_location
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Page config
st.set_page_config(
    page_title="RealEstatePRO",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 RealEstatePRO")
st.markdown("""
Search for real estate properties using natural language. Ask about:
- Properties in specific locations
- Price ranges
- Number of bedrooms/bathrooms
- Average prices in areas
""")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for API keys
with st.sidebar:
    st.header("Configuration")
    #openai_api_key = st.text_input("OpenAI API Key", type="password")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    #langchain_api_key = st.text_input("LangChain API Key", type="password")
    position_api_key = st.text_input("Position API Key", type="password")
    
    if not gemini_api_key:
        st.warning("Please enter your Gemini API key")
    
    if not position_api_key:
        st.warning("Please enter your Position API key")

    #if not openai_api_key:
        #st.warning("Please enter your OpenAI API key")
    #if not langchain_api_key:
    #    st.warning("Please enter your LangChain API key")

if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    os.environ["POSITION_API_KEY"] = position_api_key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "realestatepro-475107-9bec18ee71b8.json"

# Initialize components if API keys are provided
#if openai_api_key: #and langchain_api_key:
    # Set environment variables
    #os.environ["OPENAI_API_KEY"] = openai_api_key
   # os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
   # os.environ["LANGCHAIN_TRACING_V2"] = "true"
   # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    #os.environ["LANGCHAIN_PROJECT"] = "realestatepro-streamlit"

    # Initialize components
    @st.cache_resource
    def initialize_components():
        # Load FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = FAISS.load_local(
            "faiss_index_dir", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Initialize tools
        tools = [retrieve, average_price, final_answer, calculate_travel_time, find_places_near_location]
        name2tool = {tool.name: tool.func for tool in tools}
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)
        
        # Initialize prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You're a helpful real estate assistant that will provide information about listings and "
                "assist users in finding properties based on their preferences and requirements. "
                "When answering questions, use the tools provided. After using a "
                "tool, check the output in the 'scratchpad'. If you have an answer, "
                "provide it directly to the user."
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Initialize agent
        agent = CustomAgentExecutor(prompt=prompt, llm=llm, tools=tools, name2tool=name2tool)
        return agent

    try:
        agent = initialize_components()
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("What properties are you looking for?"):
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Searching for properties..."):
                    response = agent.invoke(input=prompt)
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please enter your API keys in the sidebar to start.")

# Footer
st.markdown("---")
st.markdown("RealEstatePRO - Powered by LangChain and OpenAI")