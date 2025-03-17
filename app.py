import streamlit as st
import validators
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document  # Import Document to wrap the content properly

## Streamlit APP setup
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Groq Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def fetch_content_from_url(url):
    """Fetch content from a standard web URL using requests and BeautifulSoup."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator="\n")
    else:
        return None

def fetch_content_from_youtube(url):
    """Fetch transcript content from a YouTube video using YoutubeLoader."""
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs = loader.load()
    return docs

if st.button("Summarize the Content from YT or Website"):
    ## Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL.")
    else:
        try:
            with st.spinner("Fetching content..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    # Process YouTube URL
                    docs = fetch_content_from_youtube(generic_url)
                else:
                    # Process Web URL
                    content = fetch_content_from_url(generic_url)
                    if content:
                        # Wrap the content in a Document object
                        doc = Document(page_content=content)
                        docs = [doc]
                    else:
                        st.error("No content retrieved from the provided URL. Please check the URL and try again.")
                        docs = None

                if docs:
                    with st.spinner("Summarizing..."):
                        ## Chain for Summarization
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        output_summary = chain.run(docs)
                        st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception occurred: {e}")