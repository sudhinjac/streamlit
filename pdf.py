import streamlit as st
import subprocess
import fitz  # PyMuPDF for PDF handling
import tempfile

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    
    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        # Write the uploaded file to the temporary file
        tmp_file.write(pdf_file.read())
        tmp_file.flush()  # Ensure all data is written

        # Open the temporary PDF file
        with fitz.open(tmp_file.name) as doc:
            for page in doc:
                text += page.get_text()
    
    return text

# Function to run LLaMA model with a prompt for summarizing
def run_llama_for_summary(prompt):
    try:
        # Use subprocess to run the LLaMA model
        process = subprocess.Popen(
            ["ollama", "run", "llama3.1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(prompt)
        
        # Check for output
        if error:
            st.error(f"Error running LLaMA model: {error.strip()}")
        return output.strip() if output else "No output received."
    except Exception as e:
        st.error(f"Exception occurred while processing: {e}")
        return ""

# Streamlit app layout
st.title("PDF Document Summarizer using LLaMA 3.1")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.write("Extracted Text:")
    st.write(pdf_text[:1000])  # Display the first 1000 characters of text

    # Summarize the extracted text
    if st.button("Generate Summary"):
        summary = run_llama_for_summary(f"Summarize the following document:\n\n{pdf_text}")
        st.subheader("Summary:")