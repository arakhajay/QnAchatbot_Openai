import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Streamlit app title
st.title("Q&A Chatbot with OPENAI")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your API Key", type="password")
llm_model = st.sidebar.selectbox("Select LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=1000, value=100, step=10)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions and help with tasks."),
    ("user", "Question: {question} "),
])

# Function to generate response
def generate_response(question, api_key, llm_model, temperature, max_tokens):
    llm = ChatOpenAI(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key  # ✅ Pass the key here
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"question": question})

# Main input section
st.write("Enter your question below and click the button to get the answer.")
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not question.strip():
        st.error("Please enter a valid question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = generate_response(question, api_key, llm_model, temperature, max_tokens)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
