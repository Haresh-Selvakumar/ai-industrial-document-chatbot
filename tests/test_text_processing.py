import io
import sys
import types
import unittest
from pathlib import Path


# Lightweight stubs allow text-processing tests to run before ML packages are installed.
if "faiss" not in sys.modules:
    sys.modules["faiss"] = types.SimpleNamespace(Index=object)
if "fitz" not in sys.modules:
    sys.modules["fitz"] = types.SimpleNamespace()
if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace()
if "sentence_transformers" not in sys.modules:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = object
    sys.modules["sentence_transformers"] = module
if "transformers" not in sys.modules:
    module = types.ModuleType("transformers")
    module.AutoModelForSeq2SeqLM = object
    module.AutoTokenizer = object
    sys.modules["transformers"] = module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from prompts import build_prompt
from rag_pipeline import IndustrialRAG, TextChunk, chunk_text, clean_text, extract_document


class TextProcessingTests(unittest.TestCase):
    def test_clean_text_removes_repeated_whitespace(self):
        self.assertEqual(clean_text("Motor   voltage\n415 V"), "Motor voltage 415 V")

    def test_chunk_text_creates_overlap_and_respects_limit(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=120, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 120 for chunk in chunks))

    def test_txt_extraction(self):
        pages = extract_document(io.BytesIO(b"Rated voltage: 415 V"), "manual.txt")
        self.assertEqual(pages, [(1, "Rated voltage: 415 V")])

    def test_prompt_contains_context_question_and_guardrail(self):
        prompt = build_prompt(
            "Motor rating is 5.5 kW.",
            "What is the rating?",
            config.NOT_FOUND_MESSAGE,
        )
        self.assertIn("Motor rating is 5.5 kW.", prompt)
        self.assertIn("What is the rating?", prompt)
        self.assertIn(config.NOT_FOUND_MESSAGE, prompt)

    def _sample_pipeline(self):
        manual = (PROJECT_ROOT / "sample_documents" / "industrial_motor_manual.txt").read_text(
            encoding="utf-8"
        )
        pipeline = IndustrialRAG()
        pipeline.chunks = [
            TextChunk(text=part, source="industrial_motor_manual.txt", page=1, chunk_id=index)
            for index, part in enumerate(
                chunk_text(manual, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            )
        ]
        return pipeline

    def test_complete_safety_section_is_extracted(self):
        answer = self._sample_pipeline()._extract_operational_section(
            "What safety precautions must be followed before inspecting the motor?"
        )
        self.assertIn("lockout/tagout", answer)
        self.assertIn("verify absence of voltage", answer)
        self.assertIn("20 minutes", answer)

    def test_startup_section_is_numbered(self):
        answer = self._sample_pipeline()._extract_operational_section(
            "Explain the startup procedure."
        )
        self.assertTrue(answer.startswith("1. Clear personnel"))
        self.assertIn("START push button", answer)

    def test_overheating_checklist_is_complete(self):
        answer = self._sample_pipeline()._extract_operational_section(
            "What should be checked when the motor overheats?"
        )
        self.assertIn("mechanical overload", answer)
        self.assertIn("supply-voltage imbalance", answer)
        self.assertIn("40 °C", answer)


if __name__ == "__main__":
    unittest.main()
