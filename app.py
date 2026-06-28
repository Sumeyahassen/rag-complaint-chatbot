import streamlit as st # type: ignore
from src.retriever import load_vector_store, retrieve
from src.generator import build_prompt, generate_answer

# Set page configuration
st.set_page_config(
    page_title="CrediTrust Complaint Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI styling with custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #0e1117;
    }
    
    /* Sleek Title Gradient */
    .title-gradient {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #a3a8b4;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Expander custom styling */
    .streamlit-expanderHeader {
        background-color: #1e222b !important;
        border-radius: 8px !important;
        border: 1px solid #2e333d !important;
    }
    
    /* Source chunk card styling */
    .source-card {
        background-color: #1a1c23;
        border-left: 4px solid #92FE9D;
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache the vector store to avoid reading parquet on every interaction
@st.cache_resource
def get_cached_vector_store():
    return load_vector_store()

vector_store = get_cached_vector_store()

# Sidebar Setup
with st.sidebar:
    st.markdown("<h2 style='color:#92FE9D;'>CrediTrust</h2>", unsafe_allow_html=True)
    st.markdown("### RAG Chatbot Settings")
    st.markdown("Analyze customer complaints efficiently using Retrieval-Augmented Generation.")
    
    st.divider()
    
    # Model info / configurations
    st.markdown("**System Information**")
    st.info("""
    - **Retriever:** `all-MiniLM-L6-v2`
    - **Generator:** `flan-t5-base`
    - **Sources Retrieved (k):** `3`
    """)
    
    st.divider()
    
    # Clear conversation button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main page layout
st.markdown("<div class='title-gradient'>CrediTrust Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze customer complaints in real-time using generative AI.</div>", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are sources associated with the assistant message, show them
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("🔍 View Source Complaints"):
                for idx, chunk in enumerate(message["sources"]):
                    st.markdown(f"<div class='source-card'><b>Complaint {idx+1}:</b><br>{chunk}</div>", unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question about credit card complaints..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        # Show a spinner while retrieving and generating
        with st.spinner("Searching complaints database and generating answer..."):
            # 1. Retrieve
            chunks = retrieve(prompt, vector_store, k=3)
            # 2. Build prompt & Generate
            gen_prompt = build_prompt(chunks, prompt)
            answer = generate_answer(gen_prompt)
            
            # Display answer
            st.markdown(answer)
            
            # Display sources
            with st.expander("🔍 View Source Complaints"):
                for idx, chunk in enumerate(chunks):
                    st.markdown(f"<div class='source-card'><b>Complaint {idx+1}:</b><br>{chunk}</div>", unsafe_allow_html=True)
                    
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": chunks
        })

