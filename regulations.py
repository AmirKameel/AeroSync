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

def detect_document_type(doc):
    """Detect if the document is IOSA/ECAR/Other type."""
    first_page = doc[0].get_text()
    if re.search(r'ECAR\s+Part', first_page, re.IGNORECASE):
        return 'ecar'
    elif any(section in first_page for section in ['ORG', 'FLT', 'DSP', 'MNT', 'CAB', 'GRH', 'CGO', 'SEC']):
        return 'iosa'
    return 'other'

def parse_ecar_sections(doc):
    """Parse ECAR style documents."""
    sections = []
    current_section = None
    current_text = []
    
    # Pattern for ECAR sections like "45.1 Nationality and registration marks: General"
    section_pattern = r'^(\d+\.\d+)\s+(.+)$'
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(section_pattern, line)
            if match:
                # Save previous section if exists
                if current_section:
                    sections.append({
                        'title': current_section,
                        'page': page_num,
                        'text': '\n'.join(current_text),
                        'subsections': parse_small_subsections('\n'.join(current_text))
                    })
                
                current_section = line
                current_text = []
            else:
                current_text.append(line)
    
    # Add last section
    if current_section:
        sections.append({
            'title': current_section,
            'page': page_num,
            'text': '\n'.join(current_text),
            'subsections': parse_small_subsections('\n'.join(current_text))
        })
    
    return sections

def is_valid_header(line, page_text, line_index, lines, page, page_num):
    """
    Validate if a line is a genuine header like ORG 1.1.1 that appears in bold
    or ECAR style header.
    """
    # Skip empty lines
    line = line.strip()
    if not line:
        return False
    
    # Check for ECAR style headers (e.g., "45.1 Title")
    ecar_pattern = r'^\d+\.\d+\s+[A-Za-z]'
    if re.match(ecar_pattern, line):
        return True
    
    # Original IOSA style header check
    header_pattern = r'^(ORG|FLT|DSP|MNT|CAB|GRH|CGO|SEC)\s*(\d+(?:\.\d+)*)$'
    match = re.match(header_pattern, line)
    
    if not match:
        return False

    # Check if the section is within its valid page range for IOSA
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
                    if span.get("flags", 0) & 2**2:  # Check for bold flag
                        return True
    return False

def extract_toc_and_special_sections(pdf_path, expand_pages=8):
    """Extract TOC and special sections with support for different document types."""
    doc = fitz.open(pdf_path)
    sections = []
    
    # Detect document type
    doc_type = detect_document_type(doc)
    
    if doc_type == 'ecar':
        # Use ECAR specific parsing
        sections = parse_ecar_sections(doc)
    else:
        # Original parsing logic for IOSA and other documents
        toc = doc.get_toc()
        
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

        # Second pass: Find special sections
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

# Rest of your original code remains unchanged
def extract_section_text(doc, start_page, header, expand_pages=8):
    """Extract text for a section with improved boundary detection."""
    section_text = ""
    header_pattern = r'^\s*((?:ORG|FLT|DSP|MNT|CAB|GRH|CGO|SEC)\s+\d+(?:\.\d+)*|\d+\.\d+\s+[A-Za-z].*?)\s*$'
    
    for i in range(start_page, min(start_page + expand_pages, len(doc))):
        page = doc.load_page(i)
        page_text = page.get_text("text")
        lines = page_text.split('\n')
        
        for line_index, line in enumerate(lines):
            if is_valid_header(line, page_text, line_index, lines, page, i) and header not in line:
                return section_text.strip()
            section_text += line + "\n"
            
    return section_text.strip()


def extract_section_with_gpt(section_name, chunk_text):
    """Extract specific section using GPT."""
    model_id = 'gpt-3.5-turbo'

    # Load the secrets from the toml file
    api_key = st.secrets["OPEN_AI_KEY"]

    # Create the OpenAI client using the API key from secrets.toml
    openai.api_key = api_key

    response = openai.chat.completions.create(
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


# Original parse_small_subsections function remains unchanged
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

# Your original main function and GPT integration remain unchanged
def main():
    st.title("AeroSync Regulations Parser")

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
