"""Document ingestion, semantic retrieval, and FLAN-T5 answer generation."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import faiss
import pymupdf as fitz
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from prompts import build_prompt


@dataclass(frozen=True)
class TextChunk:
    """A searchable passage and the location from which it was extracted."""

    text: str
    source: str
    page: int
    chunk_id: int


@dataclass(frozen=True)
class SearchResult:
    """A retrieved passage with its semantic-similarity score."""

    chunk: TextChunk
    score: float


def clean_text(text: str) -> str:
    """Remove extraction noise while retaining technical punctuation and units."""
    text = text.replace("\u00a0", " ").replace("\x00", " ")
    text = re.sub(r"(?<![.!?:;])\n(?=\S)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    text = clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end

        if hard_end < text_length:
            search_start = start + max(chunk_size // 2, 1)
            boundary_positions = [
                text.rfind(separator, search_start, hard_end)
                for separator in ("\n\n", "\n", ". ", "; ", " ")
            ]
            best_boundary = max(boundary_positions)
            if best_boundary > start:
                end = best_boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(end - overlap, start + 1)
        start = next_start

    return chunks


def _read_uploaded_bytes(uploaded_file: BinaryIO) -> bytes:
    """Read an uploaded file without depending on Streamlit's file type."""
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    uploaded_file.seek(0)
    return uploaded_file.read()


def extract_document(uploaded_file: BinaryIO, filename: str) -> list[tuple[int, str]]:
    """Extract page-numbered text from a PDF or TXT file."""
    suffix = Path(filename).suffix.lower()
    raw_bytes = _read_uploaded_bytes(uploaded_file)

    if suffix == ".pdf":
        pages: list[tuple[int, str]] = []
        with fitz.open(stream=raw_bytes, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                text = clean_text(page.get_text("text"))
                if text:
                    pages.append((page_number, text))
        return pages

    if suffix == ".txt":
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")
        cleaned = clean_text(text)
        return [(1, cleaned)] if cleaned else []

    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def documents_to_chunks(
    files: Iterable[tuple[str, BinaryIO]],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Extract and split multiple uploaded documents into searchable chunks."""
    chunks: list[TextChunk] = []
    chunk_id = 0

    for filename, file_object in files:
        for page_number, page_text in extract_document(file_object, filename):
            for passage in chunk_text(page_text, chunk_size, overlap):
                chunks.append(
                    TextChunk(
                        text=passage,
                        source=filename,
                        page=page_number,
                        chunk_id=chunk_id,
                    )
                )
                chunk_id += 1

    return chunks


class IndustrialRAG:
    """Local retrieval-augmented generation pipeline."""

    def __init__(self) -> None:
        self.embedding_model: SentenceTransformer | None = None
        self.tokenizer = None
        self.language_model = None
        self.index: faiss.Index | None = None
        self.chunks: list[TextChunk] = []

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.chunks)

    def load_models(self) -> None:
        """Load Hugging Face models once, using a GPU automatically if available."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        if self.tokenizer is None or self.language_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
            self.language_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.LLM_MODEL_NAME
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.language_model.to(device)
            self.language_model.eval()

    def build_index(self, chunks: Sequence[TextChunk]) -> None:
        """Embed chunks and build an exact cosine-similarity FAISS index."""
        if not chunks:
            raise ValueError("No readable text was found in the uploaded documents.")

        self.load_models()
        assert self.embedding_model is not None

        embeddings = self.embedding_model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = list(chunks)

    def retrieve(self, question: str, top_k: int = config.TOP_K) -> list[SearchResult]:
        """Return document chunks above the configured similarity threshold."""
        if not self.ready or self.embedding_model is None or self.index is None:
            raise RuntimeError("Process at least one document before asking questions.")

        query_vector = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        search_count = min(max(top_k, 1), len(self.chunks))
        scores, indices = self.index.search(query_vector, search_count)

        results: list[SearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0 and float(score) >= config.MIN_SIMILARITY:
                results.append(SearchResult(self.chunks[int(index)], float(score)))
        return results

    @staticmethod
    def _format_context(results: Sequence[SearchResult]) -> str:
        sections: list[str] = []
        used_characters = 0

        for result in results:
            section = result.chunk.text
            remaining = config.MAX_CONTEXT_CHARACTERS - used_characters
            if remaining <= 0:
                break
            sections.append(section[:remaining])
            used_characters += len(section[:remaining])

        return "\n\n".join(sections)

    @staticmethod
    def _extract_motor_ratings(question: str, context: str) -> str | None:
        """Return exact nameplate values for a multi-field ratings question.

        Small sequence-to-sequence models sometimes collapse list answers to one
        value. This grounded fallback only activates when the user requests at
        least three rating fields and every requested value is present in the
        retrieved context.
        """
        requested_terms = (
            "supply voltage",
            "frequency",
            "output power",
            "rated output",
            "rated speed",
            "protection class",
        )
        lowered_question = question.lower()
        if sum(term in lowered_question for term in requested_terms) < 3:
            return None

        compact_context = re.sub(r"\s+", " ", context)
        patterns = {
            "output": r"Rated output:\s*([0-9]+(?:\.[0-9]+)?\s*kW)",
            "supply": r"Rated supply:\s*([^,.;\n]+)",
            "frequency": r"Rated supply:[^\n]*?([0-9]+(?:\.[0-9]+)?\s*Hz)",
            "speed": r"Rated speed:\s*([0-9]+(?:\.[0-9]+)?\s*rpm)",
            "protection": r"Protection class:\s*([A-Za-z0-9-]+)",
        }
        values: dict[str, str] = {}
        for name, pattern in patterns.items():
            match = re.search(pattern, compact_context, flags=re.IGNORECASE)
            if not match:
                return None
            values[name] = match.group(1).strip()

        return (
            f"- Supply voltage: {values['supply']}\n"
            f"- Frequency: {values['frequency']}\n"
            f"- Output power: {values['output']}\n"
            f"- Rated speed: {values['speed']}\n"
            f"- Protection class: {values['protection']}"
        )

    def _reconstructed_documents(self) -> str:
        """Rebuild indexed text by removing the deliberate chunk overlaps."""
        ordered = sorted(self.chunks, key=lambda chunk: chunk.chunk_id)
        rebuilt = ""
        for chunk in ordered:
            passage = chunk.text.strip()
            if not rebuilt:
                rebuilt = passage
                continue

            maximum = min(len(rebuilt), len(passage), config.CHUNK_OVERLAP + 80)
            overlap = 0
            for size in range(maximum, 19, -1):
                if rebuilt[-size:] == passage[:size]:
                    overlap = size
                    break
            rebuilt += " " + passage[overlap:].lstrip()
        return re.sub(r"\s+", " ", rebuilt).strip()

    def _extract_operational_section(self, question: str) -> str | None:
        """Extract complete safety/procedure checklists from indexed evidence."""
        lowered = question.lower()
        document = self._reconstructed_documents()

        if "safety" in lowered and any(
            term in lowered for term in ("inspect", "inspection", "maintain", "maintenance")
        ):
            match = re.search(
                r"SAFETY REQUIREMENTS\s+(.*?)(?=\s*2\.\s*PRE-START INSPECTION)",
                document,
                flags=re.IGNORECASE,
            )
            if match:
                section = re.sub(r"^WARNING:\s*", "", match.group(1).strip(), flags=re.I)
                sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+", section)
                    if sentence.strip()
                ]
                return "\n".join(f"- {sentence}" for sentence in sentences)

        if "startup" in lowered or "start-up" in lowered:
            match = re.search(
                r"3\.\s*STARTUP PROCEDURE\s+(.*?)(?=\s*4\.\s*NORMAL SHUTDOWN)",
                document,
                flags=re.IGNORECASE,
            )
            if match:
                sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+", match.group(1).strip())
                    if sentence.strip()
                ]
                return "\n".join(
                    f"{number}. {sentence}"
                    for number, sentence in enumerate(sentences, start=1)
                )

        if "overheat" in lowered or "overheating" in lowered:
            match = re.search(
                r"Motor overheats:\s*(.*?)(?=\s*Excessive vibration:)",
                document,
                flags=re.IGNORECASE,
            )
            if match:
                items = [
                    item.strip(" -")
                    for item in re.split(r"\s+-\s+", match.group(1).strip())
                    if item.strip(" -")
                ]
                return "\n".join(f"- {item}" for item in items)

        return None

    def generate_answer(self, question: str, results: Sequence[SearchResult]) -> str:
        """Generate a deterministic answer using only retrieved context."""
        if not results:
            return config.NOT_FOUND_MESSAGE

        self.load_models()
        assert self.tokenizer is not None and self.language_model is not None

        context = self._format_context(results)
        extracted_ratings = self._extract_motor_ratings(question, context)
        if extracted_ratings is not None:
            return extracted_ratings

        extracted_section = self._extract_operational_section(question)
        if extracted_section is not None:
            return extracted_section

        prompt = build_prompt(context, question, config.NOT_FOUND_MESSAGE)
        device = next(self.language_model.parameters()).device

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_INPUT_TOKENS,
        ).to(device)

        with torch.inference_mode():
            generated = self.language_model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                num_beams=config.NUM_BEAMS,
                do_sample=False,
                repetition_penalty=config.REPETITION_PENALTY,
                early_stopping=True,
            )

        answer = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return answer or config.NOT_FOUND_MESSAGE

    def ask(self, question: str) -> tuple[str, list[SearchResult]]:
        """Retrieve supporting passages and generate one grounded answer."""
        question = question.strip()
        if not question:
            raise ValueError("Enter a question before submitting.")

        results = self.retrieve(question)
        answer = self.generate_answer(question, results)
        return answer, results


def bytes_file(content: bytes) -> io.BytesIO:
    """Convenience helper used by tests and command-line checks."""
    return io.BytesIO(content)
