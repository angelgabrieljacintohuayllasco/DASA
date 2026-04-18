"""
build_embedding_cache.py — Pre-computa embeddings para los 66k registros de SpanishBFF.

Genera dos archivos en el directorio SHARD:
  embeddings.npy     — matriz float32 (N, 384)  ~96 MB para 66k registros
  embedding_keys.json — lista de claves en el mismo orden que las filas

En queries, Agent A carga estos archivos una sola vez, hace coseno vectorizado
en ~10ms, y recupera el registro completo de SHARD solo para el top-k.

Uso:
    python tools/build_embedding_cache.py
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, r'C:\Users\PC\Desktop\SHARD')

from datasets import load_dataset

DB_DIR = Path(r'C:\Users\PC\Desktop\DASA\data\spanish_bff_shard')
EMBED_FILE = DB_DIR / "embeddings.npy"
KEYS_FILE  = DB_DIR / "embedding_keys.json"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 512

def main():
    print("[1/3] Cargando SpanishBFF desde cache...")
    ds = load_dataset("MMG/SpanishBFF")["train"]
    print(f"      {len(ds)} registros.")

    print(f"[2/3] Cargando modelo {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"[3/3] Codificando {len(ds)} registros en batches de {BATCH_SIZE}...")
    texts = [f"{row['lemma']}. {row['definition']}" for row in ds]
    keys  = [row['lemma'] for row in ds]

    all_embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # coseno = producto punto
        convert_to_numpy=True,
    )

    print(f"\nGuardando embeddings: {EMBED_FILE}")
    np.save(str(EMBED_FILE), all_embeddings.astype(np.float32))

    print(f"Guardando claves:     {KEYS_FILE}")
    with open(KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False)

    size_mb = EMBED_FILE.stat().st_size / 1024 / 1024
    print(f"\nListo: {len(keys)} registros, {size_mb:.1f} MB")
    print(f"Para usar: DASAConfig(use_shard_backend=True, shard_db_path={str(DB_DIR)!r}, shard_num_shards=256)")

if __name__ == "__main__":
    main()
