"""Central configuration for the Industrial Document Assistant."""

APP_TITLE = "Industrial Document Assistant"
APP_SUBTITLE = "Ask grounded questions from industrial manuals and technical documents."

LLM_MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 2
MIN_SIMILARITY = 0.25
MAX_CONTEXT_CHARACTERS = 950

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 180
NUM_BEAMS = 4
REPETITION_PENALTY = 1.1

NOT_FOUND_MESSAGE = (
    "The requested information was not found in the uploaded documents."
)

SUPPORTED_FILE_TYPES = ["pdf", "txt"]
