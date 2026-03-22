import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load env
load_dotenv()

# Page config
st.set_page_config(page_title="AI Mood Chatbot", page_icon="🤖", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🤖 AI Mood Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Choose a mood & start chatting!</p>", unsafe_allow_html=True)

# Sidebar for mode selection
st.sidebar.title("🎭 Select AI Mood")

mode_option = st.sidebar.radio(
    "Choose mode:",
    ["😡 Angry", "😂 Funny", "😢 Sad"]
)

# Mode mapping
if mode_option == "😡 Angry":
    mode = "You are an angry AI Agent. You respond aggressively and impatiently."
elif mode_option == "😂 Funny":
    mode = "You are a very funny AI agent. You respond with humor and jokes."
else:
    mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."

# Initialize model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=mode)]
    st.session_state.current_mode = mode

# Reset chat if mode changes
if "current_mode" not in st.session_state or st.session_state.current_mode != mode:
    st.session_state.messages = [SystemMessage(content=mode)]
    st.session_state.current_mode = mode

# Display chat history
for msg in st.session_state.messages[1:]:  # skip system
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    response = model.invoke(st.session_state.messages)

    # Add AI message
    st.session_state.messages.append(AIMessage(content=response.content))

    with st.chat_message("assistant"):
        st.markdown(response.content)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Streamlit + LangChain</p>",
    unsafe_allow_html=True
)