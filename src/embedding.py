from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
import sys

sys.path.append(str(root_dir))
from src.config.model import SILICONFLOW_API_KEY, SILICONFLOW_API_URL
import requests
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np

"""curl Example:
curl --request POST \
  --url https://api.siliconflow.cn/v1/embeddings \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '{
  "model": "BAAI/bge-large-zh-v1.5",
  "input": "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!"
}'
"""


_LOCAL_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _resolve_local_model_path(model: str) -> Path | None:
    """Resolve potential local model path."""
    path = Path(model)
    if not path.is_absolute():
        path = (root_dir / path).resolve()
    if path.exists():
        return path
    return None


def _get_local_model(model_path: Path) -> SentenceTransformer:
    """Load and cache SentenceTransformer instances for local models."""
    key = str(model_path)
    if key not in _LOCAL_MODEL_CACHE:
        logger.info(f"Loading local embedding model from {model_path}")
        _LOCAL_MODEL_CACHE[key] = SentenceTransformer(
            key,
            trust_remote_code=True,
        )
    return _LOCAL_MODEL_CACHE[key]


def _get_remote_embedding(input_text: str, model: str):
    """Fetch embedding via SiliconFlow API."""
    url = SILICONFLOW_API_URL + "/embeddings"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"model": model, "input": input_text}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        embedding = response.json().get("data", [])[0].get("embedding", [])
        if not embedding:
            raise ValueError("No embedding found in the response.")
        return embedding
    logger.error(f"Failed to get embedding: {response.status_code} - {response.text}")
    return None


def get_embedding(input: str, model: str = "Qwen/Qwen3-Embedding-0.6B"):
    """Get embeddings from either local SentenceTransformer or remote API."""
    local_model_path = _resolve_local_model_path(model)
    if local_model_path:
        try:
            st_model = _get_local_model(local_model_path)
            emb = st_model.encode(
                [input],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            if isinstance(emb, np.ndarray):
                return emb.tolist()
            return emb
        except Exception as exc:
            logger.error(f"Local model encoding failed for {model}: {exc}")
            raise

    return _get_remote_embedding(input, model)


if __name__ == "__main__":
    test_input = "风急天高猿啸哀，渚清沙白鸟飞回，无边落木萧萧下，不尽长江滚滚来"
    embedding = get_embedding(test_input)
    print(f"Dimension of embedding: {len(embedding) if embedding else 'N/A'}")
