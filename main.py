#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Legal Notice Generator (with Groq Cloud API)

- Ingest Bare Acts / PDFs into a Chroma vector store (local embeddings).
- Retrieve top-k context and draft a structured legal notice via Groq Cloud API.
- Export the notice as a nicely formatted PDF.
- Includes error handling, metadata, and CLI utilities.

Requirements:
  pip install typer[all] python-dotenv pypdf chromadb sentence-transformers groq rich fpdf

Environment:
  .env must contain:
    GROQ_API_KEY=your_groq_api_key_here
    GROQ_MODEL=mixtral-8x7b-32768  # or llama2-70b-4096, gemma-7b-it, etc.
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from pypdf import PdfReader
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from fpdf import FPDF
from groq import Groq
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML, CSS

import chromadb
from chromadb.config import Settings  # noqa: F401  (kept for clarity)
from sentence_transformers import SentenceTransformer

# ---------- GLOBALS ----------
app = typer.Typer(help="Legal Notice Generator (with Groq Cloud API)")
console = Console()

# Load env
load_dotenv()
DB_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

# Validate Groq API key
if not GROQ_API_KEY:
    rprint("[red]Error: GROQ_API_KEY not found in environment variables.[/red]")
    rprint("[yellow]Please add GROQ_API_KEY=your_api_key to your .env file[/yellow]")
    sys.exit(1)

# Initialize Groq client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    rprint(f"[red]Failed to initialize Groq client: {e}[/red]")
    sys.exit(1)

# Initialize embedding model once
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    rprint(f"[red]Failed to load SentenceTransformer: {e}[/red]")
    sys.exit(1)


class ChromaEmbeddingFunction:
    """Bridge for Chroma to call local sentence-transformers."""
    def __call__(self, input: List[str]):
        return embedder.encode(input, convert_to_numpy=True).tolist()

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embedder.encode(texts, convert_to_numpy=True).tolist()

    def name(self):
        return "all-MiniLM-L6-v2"


# Persistent Chroma client + collection
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    name="legal-notices",
    embedding_function=ChromaEmbeddingFunction()
)

# ---------- HELPERS ----------
def info_banner():
    rprint(
        f"[bold cyan]Legal Notice Generator[/bold cyan]  "
        f"• [green]Model:[/green] {GROQ_MODEL}  "
        f"• [green]DB:[/green] {Path(DB_DIR).resolve()}"
    )


def chunk_text_words(text: str, max_words: int = 250) -> List[str]:
    """
    Simple word-based chunking that stays within small embedding sweet spots.
    Avoids external tokenizers for offline reliability.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def extract_pdf_text_chunks(pdf_path: Path, max_words: int = 250) -> List[str]:
    chunks: List[str] = []
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}") from e

    for idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            rprint(f"[yellow]Warning: failed to extract text from page {idx+1}: {e}[/yellow]")
            continue
        if not text:
            continue
        for c in chunk_text_words(text, max_words=max_words):
            chunks.append(c)
    return chunks


def ingest_pdf(pdf_path: Path, source_label: Optional[str] = None) -> int:
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"No file found at: {pdf_path}")

    chunks = extract_pdf_text_chunks(pdf_path)
    if not chunks:
        rprint(f"[yellow]No extractable text found in '{pdf_path}'. Skipping.[/yellow]")
        return 0

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": str(pdf_path), "source_label": source_label or pdf_path.stem}] * len(chunks)

    # Let Chroma compute embeddings via our embedding function (do NOT pass precomputed vectors)
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def retrieve(query: str, k: int = 4) -> List[dict]:
    """
    Returns a list of dicts: [{document, id, metadata}, ...]
    Gracefully handles empty collections.
    """
    try:
        # If empty collection, Chroma may still "work" but return empty
        results = collection.query(query_texts=[query], n_results=k)
    except Exception as e:
        raise RuntimeError(f"Chroma query failed: {e}") from e

    docs = results.get("documents") or []
    ids = results.get("ids") or []
    metas = results.get("metadatas") or []

    # results["documents"] is a list of lists (batch dimension)
    if not docs or not docs[0]:
        return []

    out = []
    for i in range(len(docs[0])):
        out.append({
            "document": docs[0][i],
            "id": ids[0][i] if ids and ids[0] else None,
            "metadata": metas[0][i] if metas and metas[0] else {}
        })
    return out


def format_context_for_prompt(hits: List[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        src = h.get("metadata", {}).get("source_label") or h.get("metadata", {}).get("source") or "unknown"
        parts.append(f"[{i}] Source: {src}\n{h['document']}")
    return "\n\n".join(parts)


def call_groq(prompt: str, max_tokens: int = 4096) -> str:
    """
    Call Groq Cloud API and return the generated text.
    Includes error handling and timeout management.
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a meticulous Indian legal assistant. Draft professional legal notices based on provided context and queries."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            temperature=0.1,  # Low temperature for consistent legal language
            top_p=0.9,
        )
        
        return chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        if "rate_limit" in str(e).lower():
            raise RuntimeError(f"Groq API rate limit exceeded: {e}")
        elif "authentication" in str(e).lower():
            raise RuntimeError(f"Groq API authentication failed. Check your API key: {e}")
        elif "quota" in str(e).lower():
            raise RuntimeError(f"Groq API quota exceeded: {e}")
        else:
            raise RuntimeError(f"Groq API error: {e}")


def structured_prompt(query: str, context_block: str) -> str:
    """
    Constrains the LLM to output a consistent, professional notice with all details.
    """
    today = datetime.now().strftime("%d %B %Y")
    return f"""You are a senior legal professional specializing in Indian law. Using the context below and the query details, draft a formal, comprehensive legal notice that meets all requirements of Indian legal practice. Include complete names, addresses, contact information, amounts, dates, and all relevant legal acts and sections with precise citations.

=== CONTEXT START ===
{context_block if context_block.strip() else "No relevant context was retrieved."}
=== CONTEXT END ===

QUERY:
{query}

MANDATORY REQUIREMENTS:
- Draft in impeccable, formal legal English suitable for Indian courts and legal practice.
- Include ALL SPECIFIC DETAILS provided in the query without exception.
- Cite specific legal acts, sections, and provisions with complete statutory references.
- Structure the notice in accordance with established legal notice formats used in India.

OUTPUT FORMAT (STRICT COMPLIANCE REQUIRED):

Date: {today}

From:
[Complete sender name and designation]
[Law firm/Company name, if applicable]
[Complete address including street, city, state, PIN code]
[Contact telephone number]
[Email address]

To:
[Complete recipient name and designation]
[Organization/Company name]
[Complete address including street, city, state, PIN code]

Subject: LEGAL NOTICE UNDER [RELEVANT ACT/SECTION] FOR [BRIEF DESCRIPTION OF CAUSE]

NOTICE UNDER SECTION [RELEVANT SECTION] OF [ACT NAME]

Dear Sir/Madam,

WHEREAS:

1. [Detailed factual background with complete chronology, parties involved, amounts, dates, and all relevant circumstances]

2. [Additional facts and circumstances as provided in the query]

AND WHEREAS:

The aforementioned acts constitute violations of the following legal provisions:

LEGAL NOTICE:

Take notice that you are hereby called upon to:

1. [Specific demand with amount, timeline, and legal basis]
2. [Additional demands as applicable]
3. [Consequences of non-compliance]

You are further notified that in the event of your failure to comply with the above demands within [specific timeframe] from the receipt of this notice, the undersigned shall be compelled to initiate appropriate legal proceedings against you under [relevant acts and sections] and seek the following reliefs:

1. [Specific reliefs sought with amounts and legal basis]
2. [Additional reliefs]

This notice is issued without prejudice to all rights and remedies available to the undersigned under law.

Please treat this as a final opportunity to resolve this matter amicably.

Dated this {today.split()[0]} day of {today.split()[1]}, {today.split()[2]}

Yours faithfully,

[Complete name]
[Designation]
[Law firm/Company name]
[Complete address]
[Contact details]

ADVOCATE/LAWYER REGISTRATION NUMBER: [If applicable]

CONSTRAINTS:
- Use ONLY the details provided in the query for names, addresses, amounts, etc.
- Cite statutory provisions with complete accuracy and full act names.
- Maintain formal legal language throughout.
- Ensure all sections are comprehensive and legally sound.

FORMAT STRICTLY AS ABOVE - NO EXTRA COMMENTARY OR MODIFICATIONS."""


class NoticePDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Legal Notice", ln=True, align="C")
        self.ln(2)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0, "R")


def save_notice_pdf(text: str, filename: Path, title: str = "Legal Notice", append_context: Optional[List[dict]] = None):
    pdf = NoticePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_title(title)
    pdf.set_author("Legal Notice Generator")

    pdf.set_font("Arial", size=12)

    # Split on blank lines to preserve section breaks a bit better
    for paragraph in text.split("\n\n"):
        for line in paragraph.split("\n"):
            pdf.multi_cell(0, 8, line)
        pdf.ln(2)

    # Optional appendix for retrieved context
    if append_context:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Appendix: Retrieved Context", ln=True)
        pdf.set_font("Arial", size=11)
        for i, h in enumerate(append_context, 1):
            src = h.get("metadata", {}).get("source_label") or h.get("metadata", {}).get("source") or "unknown"
            pdf.multi_cell(0, 8, f"[{i}] Source: {src}")
            pdf.ln(1)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 6, h["document"])
            pdf.ln(2)
            pdf.set_font("Arial", size=11)

    pdf.output(str(filename))
    rprint(f"[green]PDF saved as {filename}[/green]")


def parse_notice_for_template(text: str) -> dict:
    lines = text.split('\n')
    template_vars = {
        'logo_left_path': 'file://' + os.path.abspath('logo.png'),  # file:// for local image
        'logo_right_path': 'file://' + os.path.abspath('logo2.png'),  # right logo
        'sender_name': '',
        'sender_address': '',
        'sender_phone': '',
        'sender_email': '',
        'recipient_name': '',
        'recipient_title': '',
        'recipient_org': '',
        'recipient_address_lines': [],
        'date': '',
        'subject': '',
        'notice_points': []
    }
    
    current_section = None
    in_whereas = False
    in_legal_notice = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('Date:'):
            template_vars['date'] = line.split(':', 1)[1].strip()
        elif line.startswith('From:'):
            current_section = 'from'
        elif line.startswith('To:'):
            current_section = 'to'
        elif line.startswith('Subject:'):
            template_vars['subject'] = line.split(':', 1)[1].strip()
            current_section = None
        elif line.startswith('NOTICE UNDER'):
            current_section = 'notice_under'
            template_vars['notice_points'].append(line)
        elif line.startswith('Dear Sir/Madam,'):
            current_section = 'greeting'
        elif line.startswith('WHEREAS:'):
            in_whereas = True
            current_section = 'whereas'
            template_vars['notice_points'].append('WHEREAS:')
        elif line.startswith('AND WHEREAS:'):
            current_section = 'and_whereas'
            template_vars['notice_points'].append('AND WHEREAS:')
        elif line.startswith('LEGAL NOTICE:'):
            in_legal_notice = True
            current_section = 'legal_notice'
            template_vars['notice_points'].append('LEGAL NOTICE:')
        elif line.startswith('Dated this'):
            current_section = 'dated'
            template_vars['notice_points'].append(line)
        elif line.startswith('Yours faithfully,'):
            current_section = 'signature'
        elif current_section == 'from' and line:
            # Parse From section
            if not template_vars['sender_name'] and line:
                template_vars['sender_name'] = line
            elif not template_vars['sender_address'] and line:
                template_vars['sender_address'] = line
            elif not template_vars['sender_phone'] and line:
                template_vars['sender_phone'] = line
            elif not template_vars['sender_email'] and line:
                template_vars['sender_email'] = line
        elif current_section == 'to' and line:
            # Parse To section
            if not template_vars['recipient_name'] and line:
                template_vars['recipient_name'] = line
            elif not template_vars['recipient_title'] and line:
                template_vars['recipient_title'] = line
            elif not template_vars['recipient_org'] and line:
                template_vars['recipient_org'] = line
            else:
                template_vars['recipient_address_lines'].append(line)
        elif current_section == 'signature' and line:
            template_vars['notice_points'].append('Signature: ' + line)
            current_section = 'signature_details'
        elif current_section == 'signature_details' and line:
            template_vars['notice_points'][-1] += ' ' + line
        elif (in_whereas or in_legal_notice) and line and not line.startswith(('1.', '2.', '3.', 'Dated', 'Yours')):
            if template_vars['notice_points']:
                template_vars['notice_points'][-1] += ' ' + line
        elif current_section in ['notice_under', 'whereas', 'and_whereas', 'legal_notice', 'dated'] and line:
            if line.startswith(('1.', '2.', '3.')) or line:
                if template_vars['notice_points']:
                    template_vars['notice_points'][-1] += ' ' + line
                else:
                    template_vars['notice_points'].append(line)
    
    return template_vars


def generate_styled_pdf(template_vars: dict, output_file: Path):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template.html')
    html_content = template.render(template_vars)
    css = CSS(filename='style.css')
    HTML(string=html_content).write_pdf(str(output_file), stylesheets=[css])
    rprint(f"[green]Styled PDF saved as {output_file}[/green]")


def ensure_non_empty_store() -> bool:
    try:
        # crude heuristic: ask for 1 result on a nonsense query; if nothing, likely empty
        results = collection.query(query_texts=["__ping__"], n_results=1)
        docs = results.get("documents") or []
        return bool(docs and docs[0])
    except Exception:
        # Even if this fails, we may still have data; ignore and continue
        return True


# ---------- CLI COMMANDS ----------
@app.command()
def ingest(
    pdf: str = typer.Argument(..., help="Path to a Bare Act / PDF file"),
    label: Optional[str] = typer.Option(None, help="Optional source label to show in retrievals")
):
    """Ingest a Bare Act / PDF into vector store with metadata."""
    info_banner()
    pdf_path = Path(pdf).expanduser()
    try:
        count = ingest_pdf(pdf_path, source_label=label)
    except Exception as e:
        rprint(f"[red]Ingestion failed: {e}[/red]")
        raise typer.Exit(code=1)

    if count > 0:
        rprint(f"[green]Ingested {count} chunks from: {pdf_path}[/green]")


@app.command()
def query(
    q: str = typer.Argument(..., help="Query to draft the legal notice"),
    output: Optional[str] = typer.Option(None, help="Output PDF filename (default auto)"),
    k: int = typer.Option(4, help="Top-k context passages to retrieve"),
    include_context: bool = typer.Option(True, help="Append retrieved context as PDF appendix"),
    max_tokens: int = typer.Option(4096, help="Maximum tokens for Groq API response"),
):
    """Query legal knowledge and draft a structured notice; optionally save as PDF."""
    info_banner()

    if not ensure_non_empty_store():
        rprint("[yellow]Vector store seems empty. Ingest PDFs first.[/yellow]")
        raise typer.Exit(code=1)

    # Retrieve
    rprint("[cyan]Retrieving relevant context...[/cyan]")
    hits = retrieve(q, k=k)
    if not hits:
        rprint("[yellow]No relevant context found. Proceeding with disclaimer in the notice.[/yellow]")

    # Show small table of retrieved context
    if hits:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Source", style="green", width=40)
        table.add_column("Snippet", style="white")
        for i, h in enumerate(hits, 1):
            src = h.get("metadata", {}).get("source_label") or h.get("metadata", {}).get("source") or "unknown"
            snippet = (h["document"][:180] + "…") if len(h["document"]) > 200 else h["document"]
            table.add_row(str(i), src, snippet)
        console.print(table)

    context_block = format_context_for_prompt(hits) if hits else ""
    prompt = structured_prompt(q, context_block)

    # Generate
    rprint("[cyan]Generating legal notice with Groq...[/cyan]")
    try:
        answer = call_groq(prompt, max_tokens=max_tokens)
    except Exception as e:
        rprint(f"[red]Generation failed: {e}[/red]")
        raise typer.Exit(code=1)

    # Print draft to console
    rprint("\n[bold yellow]Draft Notice:[/bold yellow]\n")
    rprint(answer if answer else "[red]No text generated.[/red]")

    # Save PDF if requested or by default
    if output:
        out_path = Path(output).expanduser()
    else:
        safe_slug = "".join(c for c in q.strip().replace(" ", "_") if c.isalnum() or c in ("_", "-"))[:40]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = Path(f"notice_{safe_slug or 'draft'}_{timestamp}.pdf")

    if answer and answer.strip():
        try:
            template_vars = parse_notice_for_template(answer)
            generate_styled_pdf(template_vars, out_path)
            if include_context:
                # Optionally append context as a separate simple PDF or note
                rprint("[yellow]Note: Context appendix not included in styled PDF. Use --include-context=False or modify template.[/yellow]")
        except Exception as e:
            rprint(f"[red]Failed to save styled PDF: {e}[/red]")
            raise typer.Exit(code=1)
    else:
        rprint("[red]No notice text generated. PDF not saved.[/red]")


@app.command()
def test_groq():
    """Test Groq API connection and authentication."""
    info_banner()
    rprint("[cyan]Testing Groq API connection...[/cyan]")
    
    try:
        test_response = call_groq("Hello, this is a test message. Please respond with 'Groq API is working correctly.'")
        rprint(f"[green]✓ Groq API test successful![/green]")
        rprint(f"[green]Response:[/green] {test_response}")
    except Exception as e:
        rprint(f"[red]✗ Groq API test failed: {e}[/red]")
        rprint("[yellow]Please check your GROQ_API_KEY in the .env file[/yellow]")
        raise typer.Exit(code=1)


@app.command()
def reset(confirm: bool = typer.Option(False, help="Confirm wiping the collection")):
    """Remove the entire 'legal-notices' collection."""
    info_banner()
    if not confirm:
        rprint("[yellow]Add --confirm to proceed with deletion.[/yellow]")
        raise typer.Exit(code=1)

    try:
        client.delete_collection("legal-notices")
        rprint("[green]Collection deleted.[/green]")
    except Exception as e:
        rprint(f"[red]Failed to delete collection: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def stats():
    """Show basic collection stats."""
    info_banner()
    try:
        # Chroma doesn't expose count directly; rough probe
        results = collection.query(query_texts=["__stat__"], n_results=10)
        approx = len(results.get("ids", [[]])[0]) if results.get("ids") else 0
        rprint(f"[green]Approx sample size from query:[/green] {approx} (not exact)")
        rprint("[cyan]Note:[/cyan] Chroma client does not expose a precise count; "
               "use external store listing if needed.")
    except Exception as e:
        rprint(f"[red]Stats query failed: {e}[/red]")


# ---------- ENTRY ----------
if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        rprint("\n[red]Interrupted by user.[/red]")