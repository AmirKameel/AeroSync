import streamlit as st
import fitz  # PyMuPDF
import re
import openai
import toml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract TOC and scan for sections not in TOC
def extract_toc_and_sections(pdf_path, expand_pages=7):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # Extract the Table of Contents (TOC)
    sections = {}
    
    # Create a dictionary to map TOC entries to text in the PDF
    for toc_entry in toc:
        level, title, page = toc_entry
        try:
            # Extract text from the starting page and the following pages
            section_text = ""
            for i in range(page - 1, min(page - 1 + expand_pages + 1, len(doc))):  # Expand to the next `expand_pages`
                page_text = doc.load_page(i).get_text("text")
                if not page_text:
                    page_text = doc.load_page(i).get_text("blocks")  # Try blocks if text is empty
                section_text += page_text if page_text else "Text not available for this section\n"
            
            sections[title] = {
                "level": level,
                "page": page,
                "text": section_text.strip()  # Store the combined text for the section
            }
        except Exception as e:
            sections[title] = {
                "level": level,
                "page": page,
                "text": f"Error extracting text: {str(e)}"
            }
    
    # Function to detect section headers like "ORG 1.1.1", "ORG 2.3.4", etc.
    def find_section_headers(page_text):
        pattern = r'\b(ORG \d+(\.\d+){1,5})\b'  # Matches patterns like ORG 1.1, ORG 2.1.1, etc.
        headers = re.findall(pattern, page_text)
        return [header[0] for header in headers]

    # Scan each page for section headers not in the TOC
    for page_num in range(len(doc)):
        page_text = doc.load_page(page_num).get_text("text")
        headers = find_section_headers(page_text)
        
        for header in headers:
            # If header is not already in sections, add it
            if header not in sections:
                # Extract text for this header
                section_text = ""
                for i in range(page_num, min(page_num + expand_pages + 1, len(doc))):
                    page_text = doc.load_page(i).get_text("text")
                    if not page_text:
                        page_text = doc.load_page(i).get_text("blocks")  # Try blocks if text is empty
                    section_text += page_text if page_text else "Text not available for this section\n"
                
                sections[header] = {
                    "level": header.count('.') + 1,  # Determine level by the number of dots
                    "page": page_num + 1,
                    "text": section_text.strip()  # Store the combined text for the section
                }
    
    return sections

# Function to extract a section using GPT
def extract_section_with_gpt(section_name, chunk_text):
    model_id = 'gpt-4o'

    # Load the secrets from the toml file
    api_key = st.secrets["OPEN_AI_KEY"]

    # Create the OpenAI client using the API key from secrets.toml
    openai.api_key = api_key

    # OpenAI API request
    response = openai.chat.completions.create(
        model=model_id,
        messages=[
            {
                'role': 'system',
                'content': (
                    """
    Context:
    You are tasked with extracting sections from a document. Your focus is on finding specific sections based on their header names and extracting only the relevant portion. Ignore any unrelated text that appears before or after the specified section.And pay attention that when you select a parent branch you select all the children texts of it and if it is a child not have subchildren extract it only. And if you found a pattern that is repeated in each page its a footer remove it too like for example you found Copyright Jetstar Pacific Issue 2.1 0-9 
Quality Manual 5/01/2017 and some contnets and found Copyright Jetstar Pacific Issue 2.1 0-9 
Quality Manual 5/01/2017 then remove this pattern from all pages
                    """
                )
            },
            {
                'role': 'user',
                'content': (
                    f"""
    OBJECTIVE:
    You are provided with the full text of a document. Your task is to extract the section titled "{section_name}". The section starts with this title and ends at the conclusion of the relevant content.And pay attention that when you select a parent branch you select all the children texts of it and if it is a child not have subchildren extract it only. for ex: select the section 1 and there is 1.1 , 1.2 . 1.2.1 and so on you extract the childs of the section 1 , excluding any following sections or unrelated text.

    Here is the document text:
    {chunk_text}

    Please extract and return only the content of the section titled "{section_name}".
    And just reply with the text donot include any strings or words.
                    """
                )
            }
        ],
        max_tokens=4000  # Adjust token limit based on document size
    )

    # Return the extracted section
    return response.choices[0].message.content

# Streamlit app
def main():
    st.title("AeroSync Manual Parser ")
    
    # Upload the PDF
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract TOC and extra sections
        sections = extract_toc_and_sections("uploaded_pdf.pdf")
        
        # Display the TOC with expandable sections
        st.subheader("Table of Contents and Detected Sections")
        for section, details in sections.items():
            with st.expander(f"{'  ' * (details['level'] - 1)}{section} (Page {details['page']})"):
                if st.button(f"Read {section}", key=section):
                    st.write(f"**Extracted Text from {section}:**")
                    # Extract the section using GPT
                    extracted_text = extract_section_with_gpt(section, details['text'])
                    st.write(extracted_text)

if __name__ == "__main__":
    main()
