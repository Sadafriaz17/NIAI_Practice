import os
from pathlib import Path

import arxiv
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from the .env file

# ---------- Settings ----------
# Groq changes its model list often. If you get a "model not found" error,
# check the current names at console.groq.com and put one in your .env file as GROQ_MODEL.
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

LENGTH_OPTIONS = {
    "Short (about 50 words)": "about 50 words",
    "Medium (about 150 words)": "about 150 words",
    "Long (about 300 words)": "about 300 words",
}


# ---------- Step 1: search for papers ----------
def search_papers(query): #query is the text user typed.
    """Search arXiv and return a list of papers."""
    client = arxiv.Client() # send request to arxiv 
    search = arxiv.Search(
        query=query,
        sort_by=arxiv.SortCriterion.Relevance, # search relevant results and sort them 
    )
    papers = [] #papers will be stored in this list
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "authors": ", ".join(a.name for a in result.authors[:3]), #takes the first three authors. a.name for a in ... gets the name of each author.
                "year": result.published.year,
                "abstract": result.summary,
                "url": result.entry_id,
            }
        )
    return papers


# ---------- Step 2: build the LangChain chain ----------
def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You explain research papers in simple, clear language. "
                "Use only the information given. Do not invent results.",
            ),
            (
                "human",
                "Title: {title}\n\n"
                "Abstract: {abstract}\n\n"
                "Write a summary of this paper in {length}.",
            ),
        ]
    )
    llm = ChatGroq(model=MODEL_NAME, temperature=0)
    return prompt | llm | StrOutputParser()


# ---------- Step 3: load the stylesheet ----------
def load_css(file_name="style.css"):
    """Read style.css and add it to the page. The @import line is the first line in that file."""
    css_path = Path(__file__).parent / file_name # parent is app.py's directory, file_name is style.css
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------- Step 4: the Streamlit page ----------
st.set_page_config(page_title="Paper Summary Generator", page_icon="📄")
load_css()

st.title("Research Paper Summary Generator")
st.write("Search for a paper, pick one from the list, and get a summary.")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

# Search box
query = st.text_input("Paper name or topic", placeholder="Attention Is All You Need")

if st.button("Search papers"):
    if not query.strip():
        st.warning("Type a paper name or topic first.")
    else:
        with st.spinner("Searching arXiv..."):
            st.session_state["papers"] = search_papers(query)
            st.session_state["summary"] = None

papers = st.session_state.get("papers", [])

if papers:
    # Dropdown 1: the paper
    paper_index = st.selectbox(
        "Choose a paper",
        options=range(len(papers)),
        format_func=lambda i: f"{papers[i]['title']} ({papers[i]['year']})",
    )
    # Dropdown 2: the summary length
    length_label = st.selectbox("Summary length", list(LENGTH_OPTIONS))

    # Generate button
    if st.button("Generate summary"):
        paper = papers[paper_index]
        with st.spinner("Writing the summary..."):
            try:
                chain = build_chain()
                text = chain.invoke(
                    {
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "length": LENGTH_OPTIONS[length_label],
                    }
                )
                st.session_state["summary"] = {"paper": paper, "text": text}
            except Exception as error:
                st.session_state["summary"] = None
                st.error(f"Could not generate the summary: {error}")
elif "papers" in st.session_state:
    st.info("No papers found. Try different words.")

# Show the result
result = st.session_state.get("summary")
if result:
    with st.container(border=True):
        st.subheader(result["paper"]["title"])
        st.caption(f"{result['paper']['authors']} | {result['paper']['year']}")
        st.write(result["text"])
        st.markdown(f"[Open the paper on arXiv]({result['paper']['url']})")