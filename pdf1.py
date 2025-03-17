import streamlit as st
import fitz  # PyMuPDF
import subprocess

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_text = ""
    # Open the uploaded PDF file from BytesIO
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text() + "\n"
    return pdf_text

# Function to get summary and important points using LLaMA
def get_summary_and_points(text):
    # Splitting the text into chunks if needed
    chunks = text.split('\n\n')  # Adjust this logic as needed
    summary = []
    important_points = []

    for chunk in chunks:
        try:
            # Getting the summary
            summary_command = f'ollama run llama3.1 --prompt "Summarize the following text: {chunk}"'
            summary_output = subprocess.check_output(summary_command, shell=True).decode('utf-8').strip()
            summary.append(summary_output)

            # Getting important points
            points_command = f'ollama run llama3.1 --prompt "Extract important points from the following text: {chunk}"'
            points_output = subprocess.check_output(points_command, shell=True).decode('utf-8').strip()
            important_points.append(points_output)

        except subprocess.CalledProcessError as e:
            st.error(f"Error running LLaMA model on chunk: {e.output.decode('utf-8')}")
            continue

    return "\n\n".join(summary), "\n\n".join(important_points)

# Streamlit app
st.title("PDF Summary and Important Points Extractor")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text:")
    st.write(pdf_text)

    if st.button("Generate Summary and Important Points"):
        with st.spinner("Processing..."):
            summary, important_points = get_summary_and_points(pdf_text)
        
        st.subheader("Summary:")
        st.write(summary)

        st.subheader("Important Points from Each Page:")
        for i, points in enumerate(important_points.split('\n\n'), start=1):
            st.write(f"**Page {i}:**")
            st.write(points)