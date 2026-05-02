import streamlit as st
import requests
import time
import os


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Policy Assistant",
    layout="centered"
)

st.title("RAG Policy Assistant")
st.caption("HR-Finance-IT Policy Question Answering")

#Health Check
try:
    health = requests.get(f"{API_URL}/health",timeout=3)
    if health.status_code !=200:
        st.error("Backend is not ready")
        st.stop()
except Exception:
    st.error("Cannot connect to FastAPI backend")
    st.stop()
    
    
query = st.text_area(
    "Ask a question",
    placeholder="eg. What is the leave Policy during probation",
    height=100
)

ask =st.button("Ask")

if ask and query.strip():
    with st.spinner("Thinking...."):
        start=time.time()
        try:
            response=requests.post(
                f"{API_URL}/query",
                json={"query":query},
                timeout=120
            )
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()
            
        latency =time.time() -start
    
    if response.status_code !=200:
        st.error(response.text)
        st.stop()
        
    data=response.json()
    
    st.subheader("Answer")
    st.write(data["answer"])
    
    st.caption(f"Latency : {latency:.2f}s")
    
    