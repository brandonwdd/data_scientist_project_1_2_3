"""Chunking: by heading/overlap; output chunks parquet."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib
import json

from rag.ingestion.parsers import ParsedBlock


@dataclass
class Chunk:
    """Chunk with metadata per D4."""
    chunk_id: str
    content: str
    source_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    created_at: Optional[str] = None
    hash: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.hash:
            self.hash = hashlib.sha256(
                (self.source_id + str(self.page or "") + self.content).encode()
            ).hexdigest()[:16]


class Chunker:
    """
    Chunking: split by heading hierarchy, overlap, metadata.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        by_heading: bool = True
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.by_heading = by_heading

    def chunk_blocks(self, blocks: List[ParsedBlock]) -> List[Chunk]:
        """Convert parsed blocks into chunks with overlap."""
        chunks = []
        for b in blocks:
            for c in self._split_with_overlap(b):
                chunks.append(c)
        return chunks

    def _split_with_overlap(self, block: ParsedBlock) -> List[Chunk]:
        """Split block into chunks with overlap."""
        text = block.content
        out = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            slice_text = text[start:end]
            chunk_id = f"{block.source_id}:p{block.page or 0}:{idx}"
            c = Chunk(
                chunk_id=chunk_id,
                content=slice_text,
                source_id=block.source_id,
                page=block.page,
                section=block.section,
            )
            out.append(c)
            start = end - self.overlap if end < len(text) else len(text)
            idx += 1
        return out

    def to_records(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Export chunks for parquet (S3)."""
        return [asdict(c) for c in chunks]
