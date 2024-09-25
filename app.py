import streamlit as st
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import toml

# Load secrets from toml file for OpenAI API
secrets = toml.load('secrets.toml')
openai.api_key = secrets['openai']['api_key']



# Function to extract TOC and scan for sections not in TOC
def extract_toc_and_sections(pdf_path, expand_pages=5):
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


def perform_audit(iosa_checklist, input_text):
    model_id = 'gpt-4o'  
    # Load the secrets from the toml file
    secrets = toml.load('secrets.toml')

    # Create the OpenAI client using the API key from secrets.toml
    client = openai.OpenAI(api_key=secrets['openai']['api_key'])

    # OpenAI API request
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                'role': 'system',
                'content': (
                    """
    Context	Your role is pivotal as you conduct audits to ensure strict compliance with ISARPs. Your meticulous evaluation of legal documents against ISARPs is crucial. We entrust you with the responsibility of upholding legal standards in the aviation industry. During an audit, an operator is assessed against the ISARPs contained in this manual. To determine conformity with any standard or recommended practice, an auditor will gather evidence to assess the degree to which specifications are documented and implemented by the operator. In making such an assessment, the following information is applicable.	You're an aviation professional with a robust 20-year background in both the business and commercial sectors of the industry. Your expertise extends to a deep-rooted understanding of aviation regulations the world over, a strong grasp of safety protocols, and a keen perception of the regulatory differences that come into play internationally.
    Your experience is underpinned by a solid educational foundation and specialized professional training. This has equipped you with a thorough and detailed insight into the technical and regulatory dimensions of aviation. Your assessments are carried out with attention to detail and a disciplined use of language that reflects a conscientious approach to legal responsibilities.
    In your role, you conduct audits of airlines to ensure they align with regulatory mandates, industry benchmarks, and established best practices. You approach this task with a critical eye, paying close attention to the language used and its implications. It's your job to make sure that terminology is employed accurately in compliance with legal stipulations.
    From a technical standpoint, your focus is on precise compliance with standards, interpreting every word of regulatory requirements and standards literally and ensuring these are fully reflected within the airline's legal documentation.
    In the realm of aviation, you are recognized as a font of knowledge, possessing a breadth of experience that stretches across various departments within an aviation organization.
    Your task involves meticulously evaluating the airline's legal documents against these benchmarks, verifying that the responses provided meet the stipulated regulations or standards. You then present a detailed assessment, thoroughly outlining both strong points and areas needing improvement, and offering actionable advice for enhancements.
    Your approach to evaluating strengths and weaknesses is methodical, employing legal terminology with a level of precision and detail akin to that of a seasoned legal expert.
    Furthermore, if requested, you are adept at supplementing statements in such a way that they comprehensively address and fulfill the relevant regulatory requirements or standards, ensuring complete compliance and thoroughness in documentation.
    """
      )
            },
            {
                'role': 'user',
                'content': (
                    f"""
    OBJECTIVES:
    you are given a doc and a input text do the followings:
    The provided text is to be evaluated on a compliance scale against the requirements of the regulatory text or international standard, ranging from 0 to 10. A score of 0 indicates the text is entirely non-compliant or irrelevant to the set requirements, while a score of 10 denotes full compliance with the specified criteria.
    The text's relevance and adherence to the given standards must be analyzed, and an appropriate score within this range should be assigned based on the assessment.
    Provide a thorough justification for the assigned score. Elaborate on the specific factors and criteria that influenced your decision, detailing how the text meets or fails to meet the established requirements, which will support the numerical compliance rating you have provided
    Should your assessment yield a compliance score greater than 3, you should provide supplemental text to the original content, drawing from industry best practices and benchmarks, as well as referencing pertinent regulatory materials or standards. The supplementary text should be crafted in a human writing style, incorporating human factors principles to ensure it is clear, readable, and easily understood by crew members. It's important to note that aviation regulations emphasize ease of language and precision in communication.
    In the case where the provided text is deemed completely irrelevant, you are to utilize your expertise, industry benchmarks, best practices, and relevant regulatory references or standards to formulate a detailed exposition of processes, procedures, organizational structure, duty management, or any other facet within the aviation industry. The goal is to revise the text to achieve full compliance with the applicable legal requirements or standards.

    ISARPs: 
    {iosa_checklist}
    INPUT_TEXT: 
    {input_text}

    Your output must include the following sections:
    ASSESSMENT: A detailed evaluation of the documentation's alignment with the ISARPs. It should employ technical language and aviation terminology where appropriate.
    RECOMMENDATIONS: Specific, actionable suggestions aimed at improving compliance with ISARP standards. Maintain a formal and professional tone.
    OVERALL_COMPLIANCE_SCORE: A numerical rating (0 to 10) reflecting the documentation's overall compliance with the ISARPs.
    OVERALL_COMPLIANCE_TAG: A scoring tag indicating the overall compliance level with ISARPs.
    """
    )
            }
        ],
        max_tokens=4000
    )
    
    return response.choices[0].message.content


# Function to extract a section using GPT
def extract_section_with_gpt(section_name, chunk_text):
    model_id = 'gpt-4o'

    # Load the secrets from the toml file
    secrets = toml.load('secrets.toml')

    # Create the OpenAI client using the API key from secrets.toml
    client = openai.OpenAI(api_key=secrets['openai']['api_key'])

    # OpenAI API request
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                'role': 'system',
                'content': (
                    """
    Context:
    You are tasked with extracting sections from a document. Your focus is on finding specific sections based on their header names and extracting only the relevant portion. Ignore any unrelated text that appears before or after the specified section.And pay attention that when you select a parent branch you select all the children texts of it and if it is a child not have subchildren extract it only.
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


# Function to calculate cosine similarity
def calculate_similarity(chunk1, chunk2):
    vectorizer = TfidfVectorizer().fit_transform([chunk1, chunk2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return cosine_sim

# Streamlit app
def main():
    st.title("AeroSync")
    
    # Upload two PDFs
    uploaded_file_1 = st.file_uploader("Upload First PDF", type="pdf", key="pdf1")
    uploaded_file_2 = st.file_uploader("Upload Second PDF", type="pdf", key="pdf2")
    
    # Section containers
    selected_section_1 = None
    selected_section_2 = None
    
    # Process the first PDF
    if uploaded_file_1:
        with open("uploaded_pdf_1.pdf", "wb") as f:
            f.write(uploaded_file_1.getbuffer())
        sections_1 = extract_toc_and_sections("uploaded_pdf_1.pdf")
        st.subheader("Sections from First PDF")
        selected_section_1 = st.selectbox("Select a section from PDF 1", list(sections_1.keys()))
    
    # Process the second PDF
    if uploaded_file_2:
        with open("uploaded_pdf_2.pdf", "wb") as f:
            f.write(uploaded_file_2.getbuffer())
        sections_2 = extract_toc_and_sections("uploaded_pdf_2.pdf")
        st.subheader("Sections from Second PDF")
        selected_section_2 = st.selectbox("Select a section from PDF 2", list(sections_2.keys()))
    
    # Display selected sections' text
    if selected_section_1 and selected_section_2:
        chunk1 = sections_1[selected_section_1]['text']
        chunk2 = sections_2[selected_section_2]['text']
        
        # Apply GPT extraction
        extracted_chunk1 = extract_section_with_gpt(selected_section_1, chunk1)
        extracted_chunk2 = extract_section_with_gpt(selected_section_2, chunk2)
        
        # Show the extracted text before performing the audit
        st.write(f"**Extracted Text from Selected Section in PDF 1 ({selected_section_1}):**")
        st.text_area("PDF 1 Extracted Section Text", extracted_chunk1, height=200)
        
        st.write(f"**Extracted Text from Selected Section in PDF 2 ({selected_section_2}):**")
        st.text_area("PDF 2 Extracted Section Text", extracted_chunk2, height=200)
        
        # Compute similarity
        similarity = calculate_similarity(extracted_chunk1, extracted_chunk2)
        st.write(f"**Cosine Similarity between the extracted sections:** {similarity:.4f}")
        
        # Button to perform the audit
        if st.button("Perform Audit"):
            audit_result = perform_audit(chunk1, chunk2)
            st.write("**Audit Result:**")
            st.write(audit_result)

if __name__ == "__main__":
    main()
