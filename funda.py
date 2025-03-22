import streamlit as st
import fitz  # PyMuPDF for extracting text from PDF
import ollama

# Set up Streamlit UI
st.set_page_config(page_title="Stock Analysis App", page_icon="📈", layout="wide")

# Custom CSS for background color
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

st.title("Stock Analysis App 📈")
st.write("Upload a company report PDF to analyze key financial features and get a stock recommendation.")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# Function to analyze the report using Ollama models
def analyze_with_ollama(report_text, model="deepseek"):
    prompt = f"""
    You are an expert financial analyst. Analyze the following company report and extract key insights:
    
    - **Financial health**
    - **Growth potential**
    - **Risks / red flags**
    - **Debt situation**
    - **Market trends impacting the stock**
    - **Competitive advantages**
    - **Cash flow**
    - **Revenue**
    - **Growth**
    - **profit**
    
    Based on this, provide a final **BUY** or **WAIT** recommendation with reasoning.

    Report Content:
    {report_text}
    """

    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# File Upload
uploaded_file = st.file_uploader("Upload a company report PDF", type=["pdf"])

if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)

    # Display extracted text preview
    st.subheader("Extracted Report Text (Preview)")
    st.text_area("Report Content", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=200)

    # Choose the local model (DeepSeek or LLaMA 3.1)
    model_choice = st.selectbox("Select Model", ["deepseek-r1:32b", "llama3.1","deepseek-r1:14b"])

    if st.button("Analyze Report"):
        with st.spinner("Analyzing the report..."):
            analysis_result = analyze_with_ollama(extracted_text, model=model_choice)

            # Display recommendation
            st.subheader("AI Analysis & Recommendation")
            st.write(analysis_result)