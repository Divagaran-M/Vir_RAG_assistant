"""
ingest_pdfs.py — Batch PDF Ingestion into Qdrant

Scans all .pdf files in the DATA directory (including all subdirectories:
ACADEMICS, ADMINISTRATION, COLLAGE INFO, etc.) and indexes any that are
not already present in the Qdrant vector store.

Deduplication: checks Qdrant for points with matching 'filename' payload.
Skips files already indexed. Run repeatedly without re-indexing duplicates.

Usage:
    python ingest_pdfs.py
    python ingest_pdfs.py --force    # re-index everything (wipe and re-upload)
    python ingest_pdfs.py --folder "C:/custom/path"
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
# ── Ensure UTF-8 output on Windows consoles ───────────────────────────────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Ensure project root and APP directory are on python path ──────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(APP_DIR).resolve().parent

# Ensure working directory is APP_DIR so local storage ("data/qdrant_db") matches backend
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(APP_DIR, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from services.extractor import extract_text
from services.chunker import chunk_text
from services.embeddings import generate_document_embeddings
from services.vectordb import store_embeddings, client, COLLECTION_NAME

# Default input folder: C:\Aathi\Vir_RAG_assistant\DATA
DATA_FOLDER = PROJECT_ROOT / "DATA"
if not DATA_FOLDER.exists():
    fallback = Path(r"c:\Aathi\Vir_RAG_assistant\DATA")
    DATA_FOLDER = fallback if fallback.exists() else (Path(APP_DIR) / "data" / "uploads")

DEFAULT_DATA_FOLDER = str(DATA_FOLDER)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_indexed_filenames() -> set:
    """Return the set of 'filename' payloads already in Qdrant."""
    indexed = set()
    offset = None
    while True:
        try:
            result, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=None,
                limit=500,
                offset=offset,
                with_payload=["filename"],
                with_vectors=False,
            )
            for point in result:
                fn = point.payload.get("filename", "")
                if fn:
                    indexed.add(fn)
            if next_offset is None:
                break
            offset = next_offset
        except Exception as e:
            print(f"[Ingest] Note scrolling Qdrant: {e}")
            break
    return indexed


def _canonical_name(path: str) -> str:
    """
    Strip any UUID prefix if present:
      '073faf86006a4bacae38cd84adab334e_R2025 accadmeic regulation.pdf'
    → 'R2025 accadmeic regulation.pdf'
    """
    base = os.path.basename(path).strip()
    if len(base) > 33 and base[32] == "_":
        return base[33:].strip()
    return base


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch ingest PDFs into Qdrant from DATA directory")
    parser.add_argument("--force", action="store_true", help="Re-index all files (ignore existing)")
    parser.add_argument("--folder", default=DEFAULT_DATA_FOLDER, help=f"Path to PDF data folder (default: {DEFAULT_DATA_FOLDER})")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"[Ingest] ERROR: folder not found at '{folder}'")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"[Ingest] VIR CAMPUS ASSISTANT -- PDF INGESTION PIPELINE")
    print(f"[Ingest] Source Folder : {folder}")
    print(f"{'='*70}")

    # Recursively collect all PDF files across DATA and its subfolders
    pdf_paths = sorted(Path(folder).rglob("*.pdf"))
    print(f"[Ingest] Found {len(pdf_paths)} PDF file(s) across directory structure:")
    for p in pdf_paths:
        try:
            rel = p.relative_to(folder)
            print(f"   • {rel}")
        except Exception:
            print(f"   • {p.name}")

    if not pdf_paths:
        print("\n[Ingest] Nothing to index. Exiting.")
        return

    # ── Deduplicate by canonical name ─────────────────────────────────────────
    seen_canonical: dict[str, Path] = {}
    for pdf_path in pdf_paths:
        canon = _canonical_name(str(pdf_path))
        if canon not in seen_canonical:
            seen_canonical[canon] = pdf_path
        else:
            print(f"  [Ingest] Skipping duplicate: {pdf_path.name} (keeping {seen_canonical[canon]})")

    unique_pdfs = list(seen_canonical.items())  # [(canonical_name, path), ...]
    print(f"\n[Ingest] Unique canonical PDFs to process: {len(unique_pdfs)}")

    # ── Check what's already in Qdrant ────────────────────────────────────────
    if args.force:
        already_indexed = set()
        print("[Ingest] --force flag set: re-indexing all documents.")
    else:
        print("[Ingest] Checking Qdrant collection for already-indexed documents...")
        already_indexed = _get_indexed_filenames()
        print(f"[Ingest] Currently indexed in Qdrant ({len(already_indexed)}): {sorted(list(already_indexed))}")

    # ── Process each unique PDF ───────────────────────────────────────────────
    total_indexed = 0
    total_skipped = 0
    failed = []

    for canonical_name, pdf_path in unique_pdfs:
        print(f"\n{'-'*70}")
        print(f"[Ingest] Document: {canonical_name}")
        print(f"  Source Path : {pdf_path}")

        if canonical_name in already_indexed and not args.force:
            print(f"  [SKIP] Already indexed in Qdrant — skipping.")
            total_skipped += 1
            continue

        try:
            # 1. Extract text page by page
            pages = extract_text(str(pdf_path))
            if not pages:
                print(f"  [WARN] No text extracted — skipping (empty or scanned/image PDF).")
                total_skipped += 1
                continue
            print(f"  Pages extracted : {len(pages)}")

            # 2. Chunk text
            chunks = chunk_text(pages)
            print(f"  Chunks created  : {len(chunks)}")

            if not chunks:
                print(f"  [WARN] No chunks created — skipping.")
                total_skipped += 1
                continue

            # 3. Generate embeddings (Jina AI)
            print(f"  Generating embeddings via Jina AI ({len(chunks)} chunks)...")
            chunk_texts = [c["text"] for c in chunks]
            embeddings = generate_document_embeddings(chunk_texts)
            print(f"  Embeddings created : {len(embeddings)}")

            # 4. Store in Qdrant
            stored = store_embeddings(
                chunks=chunks,
                embeddings=embeddings,
                filename=canonical_name,
            )
            print(f"  [OK] Stored {stored} vectors into Qdrant as '{canonical_name}'")
            total_indexed += 1

        except Exception as e:
            print(f"  [FAIL] Failed to ingest '{canonical_name}': {e}")
            failed.append((canonical_name, str(e)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[Ingest] INGESTION COMPLETE")
    print(f"  Newly Indexed : {total_indexed}")
    print(f"  Skipped       : {total_skipped}")
    print(f"  Failed        : {len(failed)}")
    if failed:
        print("\n  Failed files:")
        for name, err in failed:
            print(f"    - {name}: {err}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
