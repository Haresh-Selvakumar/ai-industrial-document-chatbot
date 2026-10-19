# AI-Powered Industrial Document Chatbot

This mini project implements a local Retrieval-Augmented Generation (RAG) chatbot using Hugging Face models. It extracts technical content from PDF/TXT files, creates semantic embeddings, retrieves relevant passages with FAISS, and produces grounded answers through a Streamlit interface. A document-grounded extraction layer preserves complete technical ratings, safety instructions, operating procedures and troubleshooting checklists when a compact language model shortens a multi-part answer.

## Models

- Answer generation: `google/flan-t5-base`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval: FAISS exact cosine-similarity search

## Recommended environment

- Windows 10 or 11
- Python 3.11 or 3.12 (64-bit)
- At least 8 GB RAM
- Internet connection for the first model download
- GPU is optional

Python 3.13 is not recommended for this project because some machine-learning package wheels may lag behind the newest Python release.

## Installation on Windows

Open PowerShell inside this project folder and run:

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install 64-bit Python 3.11 and enable the Python Launcher during installation.

## Start the application

```powershell
venv\Scripts\activate
python -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## First demonstration

1. Upload `sample_documents/industrial_motor_manual.txt`.
2. Select **Process documents**.
3. Wait while the models are downloaded and the vector index is created.
4. Ask:
   - What safety precautions are required before maintenance?
   - What is the motor's rated supply?
   - Explain the startup procedure.
   - What should be checked if the motor overheats?
   - How frequently should vibration be measured?
5. Ask an unrelated question such as `Who invented the telephone?` and verify that the assistant refuses when no relevant passage is retrieved.

## Project structure

```text
AI_Industrial_Chatbot_Haresh/
|-- app.py                         Streamlit user interface
|-- config.py                      Models and retrieval settings
|-- prompts.py                     Grounded industrial prompt
|-- rag_pipeline.py                Ingestion, FAISS retrieval and generation
|-- requirements.txt               Python dependencies
|-- README.md                      Installation and usage guide
|-- .streamlit/config.toml         Light theme and stable file-watcher settings
|-- sample_documents/
|   `-- industrial_motor_manual.txt
`-- tests/
    `-- test_text_processing.py
```

## Design limitations

- Scanned image-only PDFs require OCR and are not supported in version 1.
- Generated answers must be verified against the original source for safety-critical use.
- FLAN-T5 Base is lightweight enough for a student project, but it is less capable than large hosted language models.
- The grounded extraction layer is intentionally limited to information present in indexed documents; it does not add external facts.
- The first run is slower because model files must be downloaded.

## Submission evidence to capture

- Document upload screen
- Successful knowledge-base processing status
- One factual answer with source passage
- One safety/procedure answer
- One troubleshooting answer
- One unavailable-information response

## Reports and evidence

`Project_Report/` contains the submitted PDF and DOCX reports; `Screenshots/` contains the demonstration screenshots. Source files are placed at the repository root for direct installation.

## Verification during repository preparation

The seven existing text-processing tests were rerun using the test suite's lightweight dependency stubs. These tests do not exercise actual model downloads, FAISS retrieval, generation, or the Streamlit interface. Submitted screenshots and `TEST_RESULTS.txt` are earlier evidence.
