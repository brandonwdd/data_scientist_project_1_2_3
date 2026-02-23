"""Document parsers (pymupdf / unstructured)."""

from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParsedBlock:
    """Parsed document block"""
    content: str
    source_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PdfParser:
    """Parse PDF via pymupdf (fitz)"""

    def __init__(self):
        self._fitz = None

    def _ensure_fitz(self):
        if self._fitz is None:
            try:
                import fitz  # pymupdf
                self._fitz = fitz
            except ImportError:
                raise ImportError("pymupdf required: pip install pymupdf")

    def parse(self, path: str) -> List[ParsedBlock]:
        """Parse PDF into blocks with source_id, page, section."""
        self._ensure_fitz()
        doc = self._fitz.open(path)
        blocks = []
        source_id = Path(path).stem
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                blocks.append(ParsedBlock(
                    content=text.strip(),
                    source_id=source_id,
                    page=page_num + 1,
                    section=None,
                    metadata={"path": path}
                ))
        doc.close()
        return blocks


class UnstructuredParser:
    """Parse documents via unstructured (HTML, DOCX, etc.)"""

    def __init__(self):
        self._unstructured = None

    def _ensure_unstructured(self):
        if self._unstructured is None:
            try:
                from unstructured.partition.auto import partition
                self._partition = partition
                self._unstructured = True
            except ImportError:
                raise ImportError("unstructured required: pip install unstructured")

    def parse(self, path: str) -> List[ParsedBlock]:
        """Parse via unstructured into blocks."""
        self._ensure_unstructured()
        elements = self._partition(filename=path)
        source_id = Path(path).stem
        blocks = []
        for i, el in enumerate(elements):
            text = getattr(el, "text", str(el))
            if not text.strip():
                continue
            m = getattr(el, "metadata", None) or {}
            page = m.get("page_number") if isinstance(m, dict) else getattr(m, "page_number", None)
            section = m.get("section") if isinstance(m, dict) else getattr(m, "section", None)
            blocks.append(ParsedBlock(
                content=text.strip(),
                source_id=source_id,
                page=page,
                section=section,
                metadata={"path": path, "element_index": i}
            ))
        return blocks
