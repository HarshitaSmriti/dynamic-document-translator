"""
Script to generate a comprehensive, realistic Business Requirement Document (BRD)
for testing the Dynamic Document Translator.
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color: str):
    """Sets background color of a table cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def create_sample_brd(output_path: Path):
    doc = Document()

    # Set document margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Header & Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.text = "CONFIDENTIAL - Dynamic Enterprise Solutions | Business Requirement Document"
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if hp.runs:
            hp.runs[0].font.size = Pt(8.5)
            hp.runs[0].font.color.rgb = RGBColor(128, 128, 128)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.text = "Dynamic Document Translator Project - BRD v2.4"
        fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if fp.runs:
            fp.runs[0].font.size = Pt(8.5)
            fp.runs[0].font.color.rgb = RGBColor(128, 128, 128)

    # Document Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("Business Requirement Document (BRD)")
    run_title.bold = True
    run_title.font.size = Pt(24)
    run_title.font.color.rgb = RGBColor(30, 41, 59)

    # Subtitle
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(18)
    run_sub = sub_p.add_run("Enterprise Multilingual AI Document Translation Platform")
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = RGBColor(79, 70, 229)
    run_sub.bold = True

    # Metadata Card / Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Project Name:", "Automated Multilingual Document Translation System (AMDTS)"),
        ("Document Version:", "Version 2.4 (Final Draft)"),
        ("Author & Team:", "Product Engineering & Enterprise AI Core Team"),
        ("Date & Status:", "September 24, 2026 | Approved for Implementation"),
    ]
    for idx, (label, val) in enumerate(meta_data):
        row = meta_table.rows[idx]
        cell_0, cell_1 = row.cells[0], row.cells[1]
        set_cell_background(cell_0, "F8FAFC")
        set_cell_background(cell_1, "FFFFFF")
        
        p0 = cell_0.paragraphs[0]
        r0 = p0.add_run(label)
        r0.bold = True
        r0.font.size = Pt(9.5)
        
        p1 = cell_1.paragraphs[0]
        r1 = p1.add_run(val)
        r1.font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 1. Executive Summary
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("1. Executive Summary")
    h1_run.font.color.rgb = RGBColor(30, 41, 59)
    
    p = doc.add_paragraph()
    p.add_run(
        "Modern global enterprises produce high volumes of documentation in English, Hindi, and Bengali. "
        "Manual translation is costly, error-prone, and causes significant delays in operational workflows. "
        "The objective of this project is to build an automated, high-precision document translation engine "
        "powered by fine-tuned IndicTrans2 neural models. The system must preserve complete document styling, "
        "table structures, bullet points, headers, footers, and font formatting without manual intervention."
    )

    # 2. Project Scope & Business Objectives
    h2 = doc.add_heading(level=1)
    h2.add_run("2. Project Scope and Business Objectives")

    p = doc.add_paragraph()
    p.add_run(
        "The project scope encompasses real-time and batched translation of Microsoft Word (.docx) documents. "
        "Key objectives include:"
    )

    objectives = [
        "Deliver sub-minute translation turnaround for documents up to 50 pages.",
        "Maintain over 95% semantic fidelity and BLEU score accuracy across English, Hindi, and Bengali.",
        "Preserve 100% of formatting elements including headings, bold/italic text, tables, and lists.",
        "Provide zero-leakage security with completely local, air-gapped model inference.",
    ]
    for obj in objectives:
        bp = doc.add_paragraph(style='List Bullet')
        bp.add_run(obj)

    # 3. Stakeholder & Roles
    h3 = doc.add_heading(level=1)
    h3.add_run("3. Stakeholder Analysis and Roles")

    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    hdr_titles = ["Role", "Department", "Key Responsibilities"]
    for i, title in enumerate(hdr_titles):
        set_cell_background(hdr_cells[i], "1E293B")
        p = hdr_cells[i].paragraphs[0]
        r = p.add_run(title)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

    stakeholders = [
        ("Product Manager", "Enterprise AI", "Defines business priorities, release milestones, and user experience requirements."),
        ("Lead ML Engineer", "NLP Research", "Maintains fine-tuned IndicTrans2 checkpoints and optimizes batch inference latency."),
        ("Security Officer", "Compliance", "Audits data privacy, ensuring no client document data leaves the secure on-premise boundary."),
        ("Business Operations", "Global Delivery", "Validates output document accuracy and operational cost savings across regional hubs."),
    ]
    for role, dept, resp in stakeholders:
        row = table.add_row()
        for i, text in enumerate([role, dept, resp]):
            cell = row.cells[i]
            set_cell_background(cell, "F8FAFC" if len(table.rows) % 2 == 0 else "FFFFFF")
            p = cell.paragraphs[0]
            p.add_run(text).font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 4. Functional Requirements
    h4 = doc.add_heading(level=1)
    h4.add_run("4. Functional Requirements")

    freq_table = doc.add_table(rows=1, cols=4)
    freq_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    f_headers = ["Req ID", "Feature Description", "Priority", "Acceptance Criteria"]
    for i, title in enumerate(f_headers):
        set_cell_background(freq_table.rows[0].cells[i], "4F46E5")
        p = freq_table.rows[0].cells[i].paragraphs[0]
        r = p.add_run(title)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(9.5)

    freqs = [
        ("FR-001", "Automatic Script & Language Detection", "High", "Accurately classify English, Hindi, and Bengali scripts with >90% confidence."),
        ("FR-002", "Deep Document Formatting Preservation", "Critical", "Retain all table borders, cell alignments, headers, footers, and font styles in generated DOCX."),
        ("FR-003", "Optimized Batch Inference", "High", "Process text segments with dynamic batching (batch size = 8) and fast greedy beam search."),
        ("FR-004", "Route Validation & Restriction", "Medium", "Enforce English <-> Indic routes and provide clear user notifications for unsupported Indic <-> Indic routes."),
        ("FR-005", "Interactive Web Interface", "High", "Responsive Streamlit dashboard with file drag-and-drop, progress tracking, and one-click download."),
    ]
    for rid, desc, prio, crit in freqs:
        row = freq_table.add_row()
        for i, val in enumerate([rid, desc, prio, crit]):
            cell = row.cells[i]
            set_cell_background(cell, "EEF2FF" if prio == "Critical" else ("F8FAFC" if len(freq_table.rows) % 2 == 0 else "FFFFFF"))
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if i == 2:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 5. Non-Functional Requirements
    h5 = doc.add_heading(level=1)
    h5.add_run("5. Non-Functional Requirements")

    nfrs = [
        ("NFR-01: Performance & Latency", "The translation pipeline must process a standard 10-page document within 15 seconds on modern CPU hardware."),
        ("NFR-02: Scalability & Memory Management", "Peak memory footprint during Seq2SeqLM generation must not exceed 4 GB RAM per instance."),
        ("NFR-03: Security & Privacy", "All temporary document buffers must be held in-memory via BytesIO with no persistent disk caching of sensitive data."),
        ("NFR-04: Robustness & Error Handling", "Gracefully handle corrupt Word documents, unmapped Unicode characters, and empty paragraph runs without raising unhandled exceptions."),
    ]
    for title, desc in nfrs:
        p = doc.add_paragraph()
        r_title = p.add_run(f"{title}: ")
        r_title.bold = True
        p.add_run(desc)

    # 6. Technical Architecture & Constraints
    h6 = doc.add_heading(level=1)
    h6.add_run("6. Technical Architecture and Constraints")

    p = doc.add_paragraph()
    p.add_run(
        "The system utilizes Python 3.11 with Hugging Face Transformers 4.32.1, PyTorch 2.0+, and python-docx. "
        "The primary AI model is IndicTrans2 (1B parameter distilled sequence-to-sequence model). "
        "Key architectural constraints include:"
    )

    constraints = [
        "Inference runs on CPU utilizing multi-threaded PyTorch execution threads.",
        "Custom tokenizer shims handle vocabulary mapping and sentence piece encodings without Cython toolchains.",
        "Document parsing uses non-destructive run-level text injection to preserve OpenXML markup.",
    ]
    for c in constraints:
        bp = doc.add_paragraph(style='List Bullet')
        bp.add_run(c)

    # 7. Sign-off & Approvals
    h7 = doc.add_heading(level=1)
    h7.add_run("7. Document Approval Sign-off")

    app_table = doc.add_table(rows=1, cols=4)
    app_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    app_headers = ["Approver Name", "Title", "Approval Status", "Date"]
    for i, title in enumerate(app_headers):
        set_cell_background(app_table.rows[0].cells[i], "1E293B")
        p = app_table.rows[0].cells[i].paragraphs[0]
        r = p.add_run(title)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(9.5)

    approvers = [
        ("Sarah Jenkins", "VP of Engineering", "Approved", "2026-09-24"),
        ("Dr. Rajesh Sharma", "Principal AI Architect", "Approved", "2026-09-24"),
        ("Ananya Sen", "Head of Quality Assurance", "Approved", "2026-09-24"),
    ]
    for name, title, status, dt in approvers:
        row = app_table.add_row()
        for i, val in enumerate([name, title, status, dt]):
            cell = row.cells[i]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            if i == 2:
                r.bold = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"Sample BRD created successfully at: {output_path}")

if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent / "sample_brd_document.docx"
    create_sample_brd(out_file)
