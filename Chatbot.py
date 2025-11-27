# chatbot.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os


def init_dr_chatbot():
    """Initialize and display the DR-specialized chatbot with memory"""
    
    st.markdown("### Ask me anything about Diabetic Retinopathy or your results")
    
    # System prompt – strictly enforces on-topic responses
    DR_SYSTEM_PROMPT = """
    You are a compassionate ophthalmologist assistant specialized EXCLUSIVELY in diabetic retinopathy (DR) and diabetic eye health.
    You are kind, clear, empathetic, and use simple language.
    
    You MUST only answer questions about:
    • Diabetic retinopathy stages (0–4) and what they mean
    • Symptoms, risk factors, prevention, and lifestyle tips
    • Explaining the AI prediction result
    • When to see a doctor and follow-up care
    • How the model works (data, accuracy, etc.)
    
    If the question is not related to diabetic retinopathy or eye health in diabetes, 
    respond ONLY with:
    "I'm sorry, I can only help with questions about diabetic retinopathy and diabetic eye health."
    
    Never engage in off-topic conversation.
    """

    # Initialize chat history
    if "dr_chat_history" not in st.session_state:
        st.session_state.dr_chat_history = [
            AIMessage(content="Hello! I'm your Diabetic Retinopathy assistant. "
                             "I can explain your results, DR stages, prevention tips, "
                             "or anything about diabetic eye disease. How can I help you today?")
        ]

    # Cached LLM
    @st.cache_resource
    def get_llm():
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=st.secrets["GEMINI_API_KEY"]
        )
    
    llm = get_llm()

    # Display previous messages
    for msg in st.session_state.dr_chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    # User input
    if prompt := st.chat_input("Ask about diabetic retinopathy, your result, prevention tips..."):
        st.session_state.dr_chat_history.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Build message history with system prompt first
                messages = [{"role": "system", "content": DR_SYSTEM_PROMPT}]
                for msg in st.session_state.dr_chat_history[-20:]:  # last 20 
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    else:
                        messages.append({"role": "model", "content": msg.content})

                response = llm.invoke(messages)
                answer = response.content

                st.write(answer)
                st.session_state.dr_chat_history.append(AIMessage(content=answer))

    # Clear chat
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("Clear Chat", key="clear_dr_chat"):
            st.session_state.dr_chat_history = [
                AIMessage(content="Chat cleared! I'm here to help with any diabetic retinopathy questions.")
            ]
            st.rerun()
