import streamlit as st
import fitz  # PyMuPDF
import re
import openai
import toml

def get_valid_page_range(section_type):
    """Return the valid page range for a given section type."""
    ranges = {
        'ORG': (52, 114),
        'FLT': (114, 299),
        'DSP': (299, 403),
        'MNT': (403, 490),
        'CAB': (490, 558),
        'GRH': (558, 620),
        'CGO': (620, 656),
        'SEC': (656, 700)
    }
    return ranges.get(section_type)

def is_valid_header(line, page_text, line_index, lines, page, page_num):
    """
    Validate if a line is a genuine header like ORG 1.1.1 that appears in bold
    and is within its valid page range.
    """
    # Skip empty lines
    line = line.strip()
    if not line:
        return False
    
    # Header pattern for ORG, FLT, etc followed by numbers with no other text
    header_pattern = r'^(ORG|FLT|DSP|MNT|CAB|GRH|CGO|SEC)\s*(\d+(?:\.\d+)*)$'
    match = re.match(header_pattern, line)
    
    if not match:
        return False

    # Check if the section is within its valid page range
    section_type = match.group(1)
    valid_range = get_valid_page_range(section_type)
    if not valid_range or not (valid_range[0] <= page_num + 1 <= valid_range[1]):
        return False

    # Get text with formatting information
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        for line_block in block.get("lines", []):
            for span in line_block.get("spans", []):
                span_text = span.get("text", "").strip()
                if span_text == line:
                    # Check if text is bold (font flags bit 2 indicates bold)
                    if span.get("flags", 0) & 2**2:  # Check for bold flag
                        return True
    return False


def parse_small_subsections(text):
    """Parse subsections like 1, 1.1, a), i), etc."""
    pattern = r'(?:(?:^|\n)([0-9]+(?:\.[0-9]+)*|[a-zA-Z]\)|[ivxlc]+)\s*)'
    matches = re.split(pattern, text)

    parsed_sections = []
    for i in range(1, len(matches), 2):
        title = matches[i].strip()
        content = matches[i + 1].strip() if (i + 1) < len(matches) else ""
        if title:
            parsed_sections.append({'title': title, 'content': content})
    return parsed_sections

def extract_toc_and_special_sections(pdf_path, expand_pages=8):
    """Extract TOC and special sections like ORG, FLT with improved header detection."""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    sections = []
    
    # First pass: Get TOC sections
    for toc_entry in toc:
        level, title, page = toc_entry
        section_text = extract_section_text(doc, page - 1, title, expand_pages)
        sections.append({
            'title': title,
            'level': level,
            'page': page,
            'text': section_text,
            'subsections': parse_small_subsections(section_text),
        })

    # Second pass: Find special sections with improved validation
    seen_headers = set()
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        lines = page_text.split('\n')
        
        for line_index, line in enumerate(lines):
            if is_valid_header(line, page_text, line_index, lines, page, page_num):
                header = line.strip()
                if header not in seen_headers:
                    seen_headers.add(header)
                    section_text = extract_section_text(doc, page_num, header, expand_pages)
                    sections.append({
                        'title': header,
                        'page': page_num + 1,
                        'text': section_text,
                        'subsections': parse_small_subsections(section_text),
                    })

    # Sort sections by page number
    sections.sort(key=lambda x: x['page'])
    doc.close()
    return sections




def extract_section_text(doc, start_page, header, expand_pages=4):
    """Extract text for a section with improved boundary detection."""
    section_text = ""
    header_pattern = r'^\s*((?:ORG|FLT|DSP|MNT|CAB|GRH|CGO|SEC)\s+\d+(?:\.\d+)*)\s*$'
    
    for i in range(start_page, min(start_page + expand_pages, len(doc))):
        page = doc.load_page(i)
        page_text = page.get_text("text")
        lines = page_text.split('\n')
        
        # Process each line
        for line_index, line in enumerate(lines):
            # If we find a new valid header different from our current one
            if is_valid_header(line, page_text, line_index, lines, page, i) and header not in line:
                return section_text.strip()
            section_text += line + "\n"
            
    return section_text.strip()

def extract_section_with_gpt(section_name, chunk_text):
    """Extract specific section using GPT."""
    model_id = 'gpt-3.5-turbo'

    secrets = toml.load('secrets.toml')
    client = openai.OpenAI(api_key=secrets['openai']['api_key'])

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                'role': 'system',
                'content': (
                    "You are an assistant skilled in parsing hierarchical documents. "
                    "Identify and extract a specific section and all its nested subsections."
                )
            },
            {
                'role': 'user',
                'content': (
                    f"Extract the section titled '{section_name}' and all relevant nested subsections. "
                    f"Here is the document text:\n\n{chunk_text}"
                )
            }
        ],
        max_tokens=4000,
    )
    return response.choices[0].message.content

def main():
    st.title("Regulations PDF Parser")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        sections = extract_toc_and_special_sections("uploaded_pdf.pdf")

        st.subheader("Parsed Sections and Subsections")
        search_query = st.text_input("Search Sections by Name")
        filtered_sections = [s for s in sections if search_query.lower() in s['title'].lower()]

        for idx, section in enumerate(filtered_sections):
            title = section['title']
            page = section['page']
            text = section['text']

            with st.expander(f"{title} (Page {page})"):
                if st.button(f"Extract '{title}'", key=f"extract_{idx}"):
                    gpt_content = extract_section_with_gpt(title, text)
                    st.write("Extracted Content:")
                    st.write(gpt_content)

if __name__ == "__main__":
    main()