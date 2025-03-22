import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import ollama

def extract_text_from_pdf(pdf_file):
    """Extracts text from the uploaded PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def analyze_with_ollama(text, model="llama3.1"):
    """Analyzes extracted text using a local Ollama model."""
    prompt = f"""
    Analyze the following company report and extract key insights:
    1. Financial Strength
    2. Growth Potential
    3. Risks and Red Flags
    4. Investment Recommendation (Buy or Wait)
    Report:
    {text}
    """
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Streamlit UI Setup
st.set_page_config(page_title="Stock Analysis App", page_icon="📈", layout="wide")

# Add custom CSS for background color
st.markdown(
    """
    <style>
    body {
        background-color: #FFD700;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("📊 Stock Analysis using AI")
st.write("Upload a company's financial report in PDF format to analyze its investment potential.")

# File Upload
uploaded_file = st.file_uploader("Upload PDF Report", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
    
    st.text_area("Extracted Text", extracted_text[:1000] + "...", height=200)
    
    model_choice = st.radio("Select Analysis Model", ["deepseek-r1:32b", "llama3.1","deepseek-r1:14b"], index=1)
    
    if st.button("Analyze Report"):
        with st.spinner("Analyzing report using AI..."):
            analysis_result = analyze_with_ollama(extracted_text, model=model_choice)
        st.subheader("🔍 Analysis Results")
        st.write(analysis_result)
