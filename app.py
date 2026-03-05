import os
import sys
import re
import json
import hashlib
from pathlib import Path
from collections import Counter
from urllib import request, error
# ── GPU / CUDA path fix ────────────────────────────────────────────────────────
# Fix CUDA_PATH if it's incorrectly set to the bin directory instead of the root.
# llama-cpp-python expects CUDA_PATH to point to the CUDA installation root,
# and it appends "\\bin" internally. If CUDA_PATH already ends with "\\bin",
# it results in a double "\\bin\\bin" path which causes import failures.
_cuda_path = os.environ.get("CUDA_PATH", "")
if _cuda_path:
    # Normalize path separators
    _cuda_path_normalized = _cuda_path.replace("/", "\\").rstrip("\\")
    # Check if the last directory component is "bin"
    if os.path.basename(_cuda_path_normalized).lower() == "bin":
        # Strip the bin directory (the library will add it back)
        _cuda_path = os.path.dirname(_cuda_path_normalized)
        os.environ["CUDA_PATH"] = _cuda_path
    else:
        _cuda_path = _cuda_path_normalized
    
    # Verify the CUDA runtime DLL exists before proceeding
    # Check for both CUDA 12 and CUDA 11 runtime DLLs
    cuda_dll_exists = (
        os.path.isfile(os.path.join(_cuda_path, "bin", "cudart64_12.dll")) or
        os.path.isfile(os.path.join(_cuda_path, "bin", "cudart64_11.dll")) or
        os.path.isfile(os.path.join(_cuda_path, "bin", "cudart64_110.dll"))
    )
    if not cuda_dll_exists:
        # CUDA_PATH points to a non-existent or incomplete installation
        os.environ.pop("CUDA_PATH", None)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (all overridable via .env)
# ─────────────────────────────────────────────────────────────────────────────
PDF_DATA_PATH     = os.getenv("PDF_DATA_PATH",     "data/")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vectorstore")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",   "sentence-transformers/all-MiniLM-L6-v2")
HF_INFERENCE_API  = os.getenv("HF_INFERENCE_API",  "Qwen/Qwen2.5-7B-Instruct")
HF_API_TIMEOUT    = int(os.getenv("HF_API_TIMEOUT", "45"))
HF_INFERENCE_TASK  = os.getenv("HF_INFERENCE_TASK", "conversational").strip() or "conversational"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free").strip()
OPENROUTER_FALLBACKS = [
    model.strip() for model in os.getenv("OPENROUTER_FALLBACKS", "google/gemma-2-9b-it:free,mistralai/mistral-7b-instruct:free").split(",") if model.strip()
]
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", str(HF_API_TIMEOUT)))
ALLOW_LOCAL_LLM_FALLBACK = os.getenv("ALLOW_LOCAL_LLM_FALLBACK", "false").lower() in {"1", "true", "yes", "on"}
LOCAL_LLM_PATH    = os.getenv("LOCAL_LLM_PATH",    "models/phi-2.Q4_K_M.gguf")
RETRIEVER_K       = int(os.getenv("RETRIEVER_K",   "3"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", str(max(RETRIEVER_K * 4, 12))))
PAGE_INDEX_TOP_K  = int(os.getenv("PAGE_INDEX_TOP_K", "6"))
HYBRID_ALPHA      = float(os.getenv("HYBRID_ALPHA", "0.65"))
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "120"))
HF_INFERENCE_FALLBACKS = [
    model.strip()
    for model in os.getenv(
        "HF_INFERENCE_FALLBACKS",
        "HuggingFaceH4/zephyr-7b-beta,microsoft/Phi-3-mini-4k-instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ).split(",")
    if model.strip()
]
HF_INFERENCE_TASK_FALLBACKS = [
    task.strip() for task in os.getenv("HF_INFERENCE_TASK_FALLBACKS", "text-generation,conversational").split(",") if task.strip()
]

VECTORSTORE_MANIFEST_FILE = "manifest.json"
PAGE_INDEX_FILE = "page_index.json"

# GPU layers: how many transformer layers to offload to GPU.
# For GTX 1650 4 GB with Phi-2 Q4_K_M (~1.7 GB) → all 32 layers fit.
# For Mistral-7B Q4_K_M (~4.1 GB) → use 20-24 layers (rest on CPU).
# Set to 0 to force CPU-only.
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "32"))

# Embedding device: "cuda" if GPU available, else "cpu"
def _embedding_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"
    except Exception:
        return "cpu"

EMBED_DEVICE = os.getenv("EMBED_DEVICE", "") or _embedding_device()

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthCare Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — cleaner look, better chat bubbles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General ── */
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stSidebar"]          { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
[data-testid="stSidebar"] *        { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #1f2937; border: 1px solid #334155;
    color: #e2e8f0; border-radius: 8px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #dc2626; border-color: #ef4444; color: white;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 4px;
}

/* ── Title ── */
h1 { color: #0f172a !important; }

/* ── Status badges ── */
.status-badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    margin: 4px 0;
    border: 1px solid transparent;
}
.badge-green  { background:#14532d; color:#dcfce7; border-color:#166534; }
.badge-blue   { background:#1e3a8a; color:#dbeafe; border-color:#1d4ed8; }
.badge-yellow { background:#78350f; color:#fef9c3; border-color:#a16207; }
.badge-red    { background:#7f1d1d; color:#fee2e2; border-color:#b91c1c; }

.status-label {
    font-size: 0.79rem;
    color: #bfdbfe !important;
    margin-top: 8px;
    margin-bottom: 2px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

/* ── Source expander ── */
[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (only pulled in when actually needed)
# ─────────────────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM


class OpenRouterLLM(LLM):
    """LangChain-compatible wrapper for OpenRouter chat completions."""

    model_id: str
    api_key: str
    timeout: float
    base_url: str = OPENROUTER_BASE_URL
    temperature: float = 0.2
    max_new_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://huggingface.co/spaces"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "Healthcare Assistant Chatbot"),
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {details[:240]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter network error: {exc.reason}") from exc

        choices = body.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content", "")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            text = str(content)

        if stop:
            for token in stop:
                if token and token in text:
                    text = text.split(token)[0]
                    break

        return text.strip()


class HFInferenceLLM(LLM):
    """Minimal LangChain-compatible wrapper around huggingface_hub InferenceClient."""

    model_id: str
    token: str
    timeout: float
    task: str = "conversational"
    temperature: float = 0.3
    max_new_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "hf-inference-client"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        client = InferenceClient(model=self.model_id, token=self.token, timeout=self.timeout)

        if self.task == "conversational":
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            text = response.choices[0].message.content if response.choices else ""
        else:
            text = client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        if stop:
            for token in stop:
                if token and token in text:
                    text = text.split(token)[0]
                    break

        return text


def _resolve_openrouter_key() -> str:
    """Resolve OpenRouter key from common secret names."""
    return (
        os.getenv("OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPENROUTER_KEY", "").strip()
        or os.getenv("OR_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM loader (no Streamlit UI calls inside cached function)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading language model…")
def _load_llm_internal():
    """
    Internal LLM loader without Streamlit UI calls.
    Returns: (llm, mode, error_message, gpu_info)
    - llm: The loaded LLM or None if failed
    - mode: "cloud", "local-gpu(nL)", or "local-cpu"
    - error_message: Error string if failed, None otherwise
    - gpu_info: Dict with GPU details for toast notification
    """
    openrouter_key = _resolve_openrouter_key()
    hf_token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    )

    cloud_errors = []

    # ── 1. OpenRouter Cloud API (preferred) ───────────────────────────────────
    if openrouter_key:
        candidate_models = [OPENROUTER_MODEL, *[m for m in OPENROUTER_FALLBACKS if m != OPENROUTER_MODEL]]
        for model_id in candidate_models:
            try:
                llm = OpenRouterLLM(
                    model_id=model_id,
                    api_key=openrouter_key,
                    timeout=OPENROUTER_TIMEOUT,
                    temperature=0.2,
                    max_new_tokens=512,
                )
                llm.invoke("Reply with exactly: ok")
                mode = "cloud-openrouter" if model_id == OPENROUTER_MODEL else f"cloud-openrouter-fallback({model_id})"
                return llm, mode, None, None
            except Exception as exc:
                cloud_errors.append(f"openrouter:{model_id}: {type(exc).__name__}")

    # ── 2. Hugging Face Cloud API (optional fallback) ─────────────────────────
    if hf_token:
        candidate_models = [HF_INFERENCE_API, *HF_INFERENCE_FALLBACKS]
        candidate_tasks = [HF_INFERENCE_TASK, *[t for t in HF_INFERENCE_TASK_FALLBACKS if t != HF_INFERENCE_TASK]]

        for model_id in candidate_models:
            for task_name in candidate_tasks:
                try:
                    llm = HFInferenceLLM(
                        model_id=model_id,
                        token=hf_token,
                        timeout=HF_API_TIMEOUT,
                        task=task_name,
                        temperature=0.3,
                        max_new_tokens=512,
                    )
                    llm.invoke("Reply with exactly: ok")
                    mode = "cloud-hf" if (model_id == HF_INFERENCE_API and task_name == HF_INFERENCE_TASK) else f"cloud-hf-fallback({model_id}|{task_name})"
                    return llm, mode, None, None
                except Exception as exc:
                    cloud_errors.append(f"hf:{model_id}|{task_name}: {type(exc).__name__}")

    cloud_error = "Cloud API unavailable."
    if cloud_errors:
        cloud_error = f"Cloud API unavailable ({'; '.join(cloud_errors)})"

    if not ALLOW_LOCAL_LLM_FALLBACK:
        if not openrouter_key and not hf_token:
            msg = (
                "No cloud LLM credentials found. Add at least one secret: "
                "`OPENROUTER_API_KEY` (recommended) or `HUGGINGFACEHUB_API_TOKEN`/`HF_TOKEN`."
            )
        elif hf_token and not openrouter_key:
            msg = (
                "Hugging Face token is present, but no configured HF model/task could be called. "
                "Try `HF_INFERENCE_API=Qwen/Qwen2.5-7B-Instruct` and keep fallback models enabled. "
                "You can also set `OPENROUTER_API_KEY` for a more reliable cloud path."
            )
        elif openrouter_key and not hf_token:
            msg = (
                "OpenRouter key is present, but all configured OpenRouter models failed. "
                "Try updating `OPENROUTER_MODEL` / `OPENROUTER_FALLBACKS` to currently available `:free` models."
            )
        else:
            msg = "No cloud LLM is currently available with the provided credentials/model configuration."

        return None, None, f"{msg}\n\nAttempt detail: {cloud_error}", None

    # ── 3. Local GGUF (optional fallback) ─────────────────────────────────────
    if not os.path.exists(LOCAL_LLM_PATH):
        return None, None, (
            f"Cloud LLM failed and local fallback is enabled, but no local model exists at `{LOCAL_LLM_PATH}`. "
            "Either disable `ALLOW_LOCAL_LLM_FALLBACK` or provide a GGUF file."
        ), None

    try:
        from langchain_community.llms import LlamaCpp

        # Detect GPU availability
        gpu_available = False
        gpu_name = ""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        n_gpu = N_GPU_LAYERS if gpu_available else 0
        gpu_info = None
        if gpu_available:
            gpu_info = {"name": gpu_name, "layers": n_gpu}

        llm = LlamaCpp(
            model_path=LOCAL_LLM_PATH,
            temperature=0.2,
            max_tokens=512,
            n_ctx=2048,          # context window (lower = faster, less memory)
            n_batch=256,         # prompt processing batch size
            n_gpu_layers=n_gpu,  # layers offloaded to GPU (0 = CPU only)
            n_threads=max(1, os.cpu_count() - 1),  # leave 1 core for OS
            verbose=False,
            f16_kv=True,         # use fp16 for key/value cache (saves VRAM)
        )
        mode = f"local-gpu({n_gpu}L)" if n_gpu > 0 else "local-cpu"
        return llm, mode, None, gpu_info

    except Exception as e:
        return None, None, f"Failed to load local model: {e}", None


def load_llm():
    """
    Wrapper that handles UI feedback for LLM loading.
    Priority order:
      1. OpenRouter (cloud, recommended)
      2. Hugging Face Inference API (cloud fallback)
      3. Local GGUF via LlamaCpp (optional, only when ALLOW_LOCAL_LLM_FALLBACK=true)
    """
    llm, mode, error, gpu_info = _load_llm_internal()
    
    # Handle errors with UI feedback
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    # Show GPU toast notification
    if gpu_info:
        st.toast(f"🚀 GPU detected: {gpu_info['name']} — offloading {gpu_info['layers']} layers", icon="✅")
    
    return llm, mode


# ─────────────────────────────────────────────────────────────────────────────
# Vector-store loader (no Streamlit UI calls inside cached function)
# ─────────────────────────────────────────────────────────────────────────────

def _discover_vectorstore_paths():
    """Discover likely FAISS vectorstore directories."""
    candidates = []

    # Explicit/common directories first
    for candidate in [
        VECTOR_STORE_PATH,
        "vectorstore",
        "vector_store",
        "VectorStore",
        "Vectorstore",
        "faiss_index",
        "index",
    ]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    # Hugging Face Spaces uploads can end up in arbitrary folders;
    # search for directories containing a matching *.faiss/*.pkl pair.
    try:
        for faiss_file in Path('.').glob('**/*.faiss'):
            parent = faiss_file.parent
            parent_str = str(parent)
            if parent_str.startswith('./.git') or '/.git/' in parent_str:
                continue
            if not (parent / f"{faiss_file.stem}.pkl").exists():
                continue
            if parent_str not in candidates:
                candidates.append(parent_str)
    except Exception:
        # Keep startup resilient; explicit candidates above are still used.
        pass

    return candidates


def _normalize_page_text(text: str) -> str:
    cleaned = (text or "").replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _tokenize_for_index(text: str) -> Counter:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", (text or "").lower())
    return Counter(tokens)


def _pdf_manifest(pdf_paths: list[Path]) -> list[dict]:
    manifest = []
    for pdf in sorted(pdf_paths):
        stat = pdf.stat()
        fingerprint = hashlib.sha256(
            f"{pdf.name}:{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8")
        ).hexdigest()[:16]
        manifest.append({
            "name": pdf.name,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "fingerprint": fingerprint,
        })
    return manifest


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _build_fallback_vectorstore(embeddings, reason: str):
    """Build a minimal fallback vectorstore so the app can still start."""
    fallback_text = (
        "Knowledge base is unavailable right now. "
        "A prebuilt vectorstore was not found and PDFs could not be processed. "
        f"Reason: {reason}"
    )
    return FAISS.from_texts(
        [fallback_text],
        embedding=embeddings,
        metadatas=[{"source": "system", "page": 0}],
    )


@st.cache_resource(show_spinner="📚 Loading knowledge base…")
def _load_vectorstore_internal():
    """
    Returns: (vector_store, page_index, source, error_message, build_info)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    pdf_paths = sorted(Path(PDF_DATA_PATH).glob("*.pdf")) if os.path.isdir(PDF_DATA_PATH) else []
    current_manifest = _pdf_manifest(pdf_paths)

    candidate_store_paths = _discover_vectorstore_paths()
    cached_load_error = None
    for store_path in candidate_store_paths:
        try:
            store_dir = Path(store_path)
            if not store_dir.exists() or not store_dir.is_dir():
                continue

            manifest_payload = _load_json(store_dir / VECTORSTORE_MANIFEST_FILE)
            if manifest_payload and manifest_payload.get("pdf_manifest") != current_manifest:
                continue

            page_index_payload = _load_json(store_dir / PAGE_INDEX_FILE)

            if (store_dir / "index.faiss").exists() and (store_dir / "index.pkl").exists():
                vector_store = FAISS.load_local(
                    str(store_dir), embeddings, allow_dangerous_deserialization=True
                )
                return vector_store, page_index_payload, "cached", None, None

            for faiss_file in store_dir.glob("*.faiss"):
                stem = faiss_file.stem
                if not (store_dir / f"{stem}.pkl").exists():
                    continue
                vector_store = FAISS.load_local(
                    str(store_dir),
                    embeddings,
                    index_name=stem,
                    allow_dangerous_deserialization=True,
                )
                return vector_store, page_index_payload, "cached", None, None
        except Exception as exc:
            cached_load_error = (
                f"Found cached index artifacts in `{store_path}` but failed to load: {exc}"
            )

    if not pdf_paths:
        error_msg = (
            f"No PDF files found in `{PDF_DATA_PATH}`. "
            "Please add at least one PDF to the data directory."
        )
        if cached_load_error:
            error_msg = f"{cached_load_error}. {error_msg}"
        error_msg += f" Checked vectorstore paths: {', '.join(candidate_store_paths)}"
        fallback_store = _build_fallback_vectorstore(embeddings, error_msg)
        return fallback_store, None, "fallback", error_msg, {"fallback": True}

    documents = []
    skipped_pdfs = []
    page_index_rows = []

    for pdf_path in pdf_paths:
        try:
            file_docs = PyPDFLoader(str(pdf_path)).load()
            if not file_docs:
                skipped_pdfs.append(f"{pdf_path.name} (no extractable pages)")
                continue

            for d in file_docs:
                cleaned_text = _normalize_page_text(d.page_content)
                if len(cleaned_text) < 40:
                    continue
                d.page_content = cleaned_text
                d.metadata["source"] = str(pdf_path)
                d.metadata["source_file"] = pdf_path.name
                documents.append(d)
                page_index_rows.append(
                    {
                        "source": str(pdf_path),
                        "source_file": pdf_path.name,
                        "page": int(d.metadata.get("page", 0)),
                        "content": cleaned_text,
                        "term_freq": dict(_tokenize_for_index(cleaned_text).most_common(120)),
                    }
                )
        except Exception as exc:
            skipped_pdfs.append(f"{pdf_path.name} ({exc})")

    if not documents:
        error_msg = (
            "No valid PDF content could be loaded. "
            "The PDF files may be corrupted, encrypted, or empty."
        )
        if skipped_pdfs:
            error_msg += f" Skipped files: {', '.join(skipped_pdfs)}"
        if cached_load_error:
            error_msg = f"{cached_load_error}. {error_msg}"
        error_msg += f" Checked vectorstore paths: {', '.join(candidate_store_paths)}"
        fallback_store = _build_fallback_vectorstore(embeddings, error_msg)
        build_info = {
            "num_pages": 0,
            "num_pdfs": 0,
            "num_chunks": 1,
            "device": EMBED_DEVICE.upper(),
            "skipped_pdfs": skipped_pdfs,
            "fallback": True,
        }
        return fallback_store, None, "fallback", error_msg, build_info

    num_pages = len(documents)
    num_pdfs = len(set(d.metadata.get('source', '') for d in documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    num_chunks = len(chunks)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

    page_index_payload = {
        "version": 1,
        "rows": page_index_rows,
        "pdf_manifest": current_manifest,
    }
    _save_json(Path(VECTOR_STORE_PATH) / PAGE_INDEX_FILE, page_index_payload)
    _save_json(Path(VECTOR_STORE_PATH) / VECTORSTORE_MANIFEST_FILE, {"pdf_manifest": current_manifest})

    build_info = {
        "num_pages": num_pages,
        "num_pdfs": num_pdfs,
        "num_chunks": num_chunks,
        "device": EMBED_DEVICE.upper(),
        "skipped_pdfs": skipped_pdfs,
    }

    return vector_store, page_index_payload, "built", None, build_info


def load_vectorstore():
    """
    Wrapper that handles UI feedback for vectorstore loading.
    Load FAISS index from disk (fast), or build it from PDFs on first run.
    Embeddings run on GPU if available, otherwise CPU.
    """
    vector_store, page_index, source, error, build_info = _load_vectorstore_internal()
    
    # Handle errors with UI feedback
    if error:
        if source == "fallback" and vector_store is not None:
            st.warning(f"⚠️ {error}")
        else:
            st.error(f"❌ {error}")
            st.stop()
    
    # Show build progress if freshly built
    if build_info:
        with st.status("🔨 Building knowledge base from PDFs…", expanded=True) as status:
            st.write("📄 Loading PDF files…")
            st.write(f"✅ Loaded {build_info['num_pages']} pages from {build_info['num_pdfs']} PDF(s)")
            if build_info.get("skipped_pdfs"):
                st.write("⚠️ Skipped unreadable PDF(s):")
                for skipped in build_info["skipped_pdfs"]:
                    st.write(f"- {skipped}")
            st.write("✂️ Splitting into chunks…")
            st.write(f"✅ Created {build_info['num_chunks']} chunks")
            st.write(f"🧮 Embedding chunks on {build_info['device']}…")
            st.write("✅ Vector store saved to disk")
            status.update(label="✅ Knowledge base ready!", state="complete")
    
    return vector_store, page_index, source


# ─────────────────────────────────────────────────────────────────────────────
# Initialise resources (cached — only runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
llm, llm_source = load_llm()
vector_store, page_index, vs_source = load_vectorstore()

class HybridPageRetriever(BaseRetriever):
    """Combines FAISS semantic retrieval with lightweight page-level lexical retrieval."""

    vector_store: any
    page_index: dict | None = None
    k: int = RETRIEVER_K
    fetch_k: int = RETRIEVER_FETCH_K
    lexical_k: int = PAGE_INDEX_TOP_K
    alpha: float = HYBRID_ALPHA

    def _page_hits(self, query: str) -> list[Document]:
        if not self.page_index or not self.page_index.get("rows"):
            return []

        query_terms = _tokenize_for_index(query)
        if not query_terms:
            return []

        scored = []
        for row in self.page_index["rows"]:
            term_freq = row.get("term_freq", {})
            score = sum(query_terms[t] * term_freq.get(t, 0) for t in query_terms)
            if score <= 0:
                continue
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        docs = []
        for score, row in scored[: self.lexical_k]:
            docs.append(
                Document(
                    page_content=row.get("content", ""),
                    metadata={
                        "source": row.get("source", "unknown"),
                        "source_file": row.get("source_file", "unknown"),
                        "page": row.get("page", 0),
                        "retrieval_score": score,
                        "retrieval_method": "page-index",
                    },
                )
            )
        return docs

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        semantic_docs = self.vector_store.max_marginal_relevance_search(
            query,
            k=self.k,
            fetch_k=self.fetch_k,
        )
        lexical_docs = self._page_hits(query)

        merged = {}
        for rank, doc in enumerate(semantic_docs):
            key = (doc.metadata.get("source", "unknown"), doc.metadata.get("page", 0), doc.page_content[:120])
            merged[key] = {
                "doc": doc,
                "semantic": max(0.0, 1.0 - (rank / max(1, self.fetch_k))),
                "lexical": 0.0,
            }

        for rank, doc in enumerate(lexical_docs):
            key = (doc.metadata.get("source", "unknown"), doc.metadata.get("page", 0), doc.page_content[:120])
            lexical_score = max(0.0, 1.0 - (rank / max(1, self.lexical_k)))
            if key in merged:
                merged[key]["lexical"] = max(merged[key]["lexical"], lexical_score)
            else:
                merged[key] = {"doc": doc, "semantic": 0.0, "lexical": lexical_score}

        ranked = sorted(
            merged.values(),
            key=lambda item: self.alpha * item["semantic"] + (1 - self.alpha) * item["lexical"],
            reverse=True,
        )
        return [item["doc"] for item in ranked[: self.k]]


# ─────────────────────────────────────────────────────────────────────────────
# RAG chain construction
# ─────────────────────────────────────────────────────────────────────────────
_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and the latest user question, "
     "rewrite the question as a fully self-contained query that can be "
     "understood without the history. "
     "Do NOT answer — only rewrite if necessary, otherwise return as-is."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

_QA_SYSTEM_PROMPT = (
    "You are a knowledgeable and empathetic healthcare assistant. "
    "Answer the user's question using ONLY the retrieved medical context below. "
    "If the context does not contain enough information, say so clearly and "
    "recommend consulting a qualified healthcare professional. "
    "Never fabricate medical information. "
    "Keep your answer clear, accurate, and concise (3–5 sentences). "
    "Always end with a brief reminder that this is for informational purposes only "
    "and is not a substitute for professional medical advice.\n\n"
    "Context:\n{context}"
)

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM_PROMPT),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

retriever = HybridPageRetriever(
    vector_store=vector_store,
    page_index=page_index,
    k=RETRIEVER_K,
    fetch_k=RETRIEVER_FETCH_K,
    lexical_k=PAGE_INDEX_TOP_K,
    alpha=HYBRID_ALPHA,
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, _CONTEXTUALIZE_PROMPT)
qa_chain                = create_stuff_documents_chain(llm, _QA_PROMPT)
rag_chain               = create_retrieval_chain(history_aware_retriever, qa_chain)

_SIMPLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly and professional healthcare assistant. "
     "Greet the user warmly and let them know they can ask any medical or health-related question."),
    ("human", "{input}"),
])
simple_chain = _SIMPLE_PROMPT | llm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_INITIAL_GREETING = (
    "Hello! 👋 I'm your **HealthCare Assistant**. "
    "I can answer medical and health-related questions based on a curated medical knowledge base. "
    "How can I help you today?"
)

_GREETINGS = frozenset({
    "hi", "hello", "hey", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "sup", "what's up", "yo",
})

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def _is_greeting(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in _GREETINGS


def _extract_string(response) -> str:
    if isinstance(response, str):
        return response.strip()
    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()


def _build_lc_history(messages: list) -> list:
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def _llm_mode_label(source: str) -> str:
    if source == "cloud-openrouter":
        return "☁️ Cloud (OpenRouter)"
    if source.startswith("cloud-openrouter-fallback("):
        model = source[len("cloud-openrouter-fallback("):-1]
        return f"☁️ OpenRouter fallback ({model})"
    if source == "cloud-hf":
        return "☁️ Cloud (HuggingFace)"
    if source.startswith("cloud-hf-fallback("):
        value = source[len("cloud-hf-fallback("):-1]
        model, task = value.split("|", 1) if "|" in value else (value, "unknown-task")
        return f"☁️ HF fallback ({model}, {task})"
    if source.startswith("local-gpu"):
        layers = source.split("(")[1].rstrip("L)") if "(" in source else "?"
        return f"🚀 Local GPU ({layers} layers offloaded)"
    return "💻 Local CPU"


def _active_llm_model(source: str) -> str:
    if source == "cloud-openrouter":
        return OPENROUTER_MODEL
    if source.startswith("cloud-openrouter-fallback("):
        return source[len("cloud-openrouter-fallback("):-1]
    if source == "cloud-hf":
        return HF_INFERENCE_API
    if source.startswith("cloud-hf-fallback("):
        value = source[len("cloud-hf-fallback("):-1]
        return value.split("|", 1)[0]
    if source.startswith("local-gpu") or source == "local-cpu":
        return os.path.basename(LOCAL_LLM_PATH)
    return "unknown"


def _embedding_model_name() -> str:
    return EMBEDDING_MODEL.split("/")[-1]


def _render_sources(sources: list):
    if not sources:
        return
    with st.expander("📄 Source documents", expanded=False):
        for i, src in enumerate(sources, 1):
            page        = src.metadata.get("page", "?")
            source_file = os.path.basename(src.metadata.get("source", "unknown"))
            st.markdown(f"**[{i}] {source_file} — page {page}**")
            snippet = src.page_content[:400]
            if len(src.page_content) > 400:
                snippet += "…"
            st.caption(snippet)
            if i < len(sources):
                st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=72)
    st.title("HealthCare Assistant")
    st.caption("Powered by RAG + LangChain")

    st.divider()

    # ── System Status ─────────────────────────────────────────────────────────
    st.subheader("🔧 System Status")

    llm_label = _llm_mode_label(llm_source)
    active_llm_model = _active_llm_model(llm_source)
    badge_cls = "badge-green" if "GPU" in llm_label or "Cloud" in llm_label else "badge-yellow"
    st.markdown('<div class="status-label">🧠 Generation LLM</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-badge {badge_cls}">{llm_label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Model: `{active_llm_model}`")

    if vs_source == "cached":
        vs_label = "📦 Cached"
    elif vs_source == "fallback":
        vs_label = "⚠️ Fallback"
    else:
        vs_label = "🔨 Freshly built"
    st.markdown('<div class="status-label">📚 Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-badge badge-blue">{vs_label}</span>',
        unsafe_allow_html=True,
    )

    embed_badge = "badge-green" if EMBED_DEVICE == "cuda" else "badge-blue"
    st.markdown('<div class="status-label">🧩 Embedding Setup</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-badge {embed_badge}">Device: {EMBED_DEVICE.upper()}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Embedding model: `{_embedding_model_name()}`")
    st.caption(f"Retriever: Hybrid (FAISS + Page Index), k={RETRIEVER_K}, fetch_k={RETRIEVER_FETCH_K}")

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": _INITIAL_GREETING}
        ]
        st.rerun()

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        import shutil
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # ── About ─────────────────────────────────────────────────────────────────
    st.subheader("ℹ️ About")
    st.markdown(
        "This chatbot answers health-related questions using a curated medical "
        "knowledge base via **Retrieval-Augmented Generation (RAG)**.\n\n"
        "**⚠️ Disclaimer:** For *informational purposes only*. "
        "Does **not** replace professional medical advice."
    )

    st.divider()
    st.caption("v2.0 · [GitHub](https://github.com/zonixt017/Healthcare-Assistant-Chatbot)")


# ─────────────────────────────────────────────────────────────────────────────
# Main chat UI
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": _INITIAL_GREETING}
    ]

st.title("🩺 HealthCare Assistant")
st.caption("Ask me anything about health, symptoms, medications, or medical conditions.")

# ── Render existing chat history ──────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            _render_sources(message["sources"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a health-related question…"):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        sources  = []
        response = ""

        try:
            history = _build_lc_history(st.session_state.messages[:-1])

            if _is_greeting(prompt) or not history:
                # Lightweight path — no retrieval needed
                with st.spinner("💬 Thinking…"):
                    raw      = simple_chain.invoke({"input": prompt})
                    response = _extract_string(raw)
            else:
                # Full RAG path
                with st.spinner("🔍 Searching knowledge base…"):
                    result   = rag_chain.invoke({"input": prompt, "chat_history": history})
                    response = _extract_string(result.get("answer", ""))
                    sources  = result.get("context", [])

                if not response:
                    response = (
                        "I'm sorry, I couldn't find relevant information in my knowledge base. "
                        "Please consult a qualified healthcare professional for this query."
                    )

        except Exception as e:
            response = (
                "⚠️ An error occurred while processing your request. "
                "Please try again or rephrase your question."
            )
            st.error(f"Error details: {e}")

        st.markdown(response)
        _render_sources(sources)

    # Persist assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })
