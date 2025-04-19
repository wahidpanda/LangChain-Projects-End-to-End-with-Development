import os
import streamlit as st
from constants import groq_key
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

# Set up the page config
st.set_page_config(
    page_title="AI Research Navigator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Groq LLM
@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0.7,
        model_name="llama3-8b-8192",
        groq_api_key=groq_key
    )

llm = load_llm()

# Sidebar for settings
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=AI+Research", width=150)
    st.title("Settings")
    research_focus = st.selectbox(
        "Research Focus Area",
        ["Deep Learning","Machine Learning","Generative AI", "Computer Vision", "NLP", "Reinforcement Learning", "Multimodal AI"]
    )
    year_filter = st.multiselect(
        "Filter by Year",
        ["2024", "2023", "2022"],
        default=["2024", "2023"]
    )
    st.divider()
    st.caption("AI Research Navigator v1.0")
    st.caption("© 2024 AI Research Labs")

# Main content area
st.title("AI Research Navigator")
st.write("Discover the latest breakthroughs in artificial intelligence research (2023-2024)")

# Search input
query = st.text_input(
    "Enter your research topic",
    placeholder="e.g. 'Recent advances in diffusion models'",
    label_visibility="visible"
)

search_clicked = st.button("Search")

# Prompt Templates
research_prompt = PromptTemplate(
    input_variables=['topic', 'focus_area', 'years'],
    template="""Provide a comprehensive analysis of recent advancements in {topic} with focus on {focus_area},
    specifically covering research from {years}. Include:
    1. Key technological breakthroughs
    2. Notable papers (prioritizing {years})
    3. Emerging applications
    4. Current limitations and challenges
    Structure your response with clear headings and bullet points."""
)

papers_prompt = PromptTemplate(
    input_variables=['overview'],
    template="""From this research overview: {overview}
    Extract the 5 most significant papers with:
    - Complete title
    - Authors and affiliations
    - Publication venue and year
    - Key contributions
    - Citation count (if available)
    Format as markdown with clear sectioning"""
)

trends_prompt = PromptTemplate(
    input_variables=['key_papers'],
    template="""Analyze these papers: {key_papers}
    Generate:
    1. Immediate research trends (next 12 months)
    2. Potential commercial applications
    3. Ethical considerations
    4. Future research directions
    Present as a structured report with clear sections"""
)

# Initialize chains
if search_clicked and query:
    with st.spinner("Analyzing latest research..."):
        # Create chains
        research_chain = LLMChain(
            llm=llm,
            prompt=research_prompt,
            output_key='overview'
        )
        
        papers_chain = LLMChain(
            llm=llm,
            prompt=papers_prompt,
            output_key='key_papers'
        )
        
        trends_chain = LLMChain(
            llm=llm,
            prompt=trends_prompt,
            output_key='future_trends'
        )
        
        # Execute chains
        result = SequentialChain(
            chains=[research_chain, papers_chain, trends_chain],
            input_variables=['topic', 'focus_area', 'years'],
            output_variables=['overview', 'key_papers', 'future_trends']
        )({
            'topic': query,
            'focus_area': research_focus,
            'years': ", ".join(year_filter)
        })
        
        # Display results
        st.subheader("Research Overview")
        st.markdown(result['overview'])
        
        st.subheader("Key Publications")
        st.markdown(result['key_papers'])
        
        st.subheader("Trend Analysis")
        st.markdown(result['future_trends'])

# Empty state
elif not search_clicked:
    st.info("""
    Welcome to AI Research Navigator
    Enter a research topic to discover the latest advancements in AI/ML
    Try searching for: "LLM quantization techniques 2024" or "Recent computer vision architectures"
    """)