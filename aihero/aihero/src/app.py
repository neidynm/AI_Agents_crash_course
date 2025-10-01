import streamlit as st
from pydantic_ai import Agent

# --- Configure Streamlit ---
st.set_page_config(page_title="FAQ Agent", page_icon="🤖", layout="centered")
st.title("🤖 FAQ Agent (Pydantic AI)")

# --- Initialize Agent (replace with your actual agent setup) ---
system_prompt = """
You are a helpful assistant that answers questions using the hybrid search system.
Keep answers concise, factual, and explain your reasoning clearly.
"""

agent = Agent(
    name="faq_agent",
    instructions=system_prompt,
    model="gpt-4o-mini",  # you can swap this with your model
)

# --- Initialize session state for chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display past messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask me anything..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.run_sync(prompt)  # synchronous for Streamlit
            response = result.data if hasattr(result, "data") else str(result)

        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
