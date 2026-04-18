"""
build_embedding_cache.py — Pre-computa embeddings para cualquier dataset SHARD.

Genera dos archivos en el directorio de la base de datos SHARD:
  embeddings.npy      — matriz float32 (N, 384)  ~96 MB para 66k registros
  embedding_keys.json — lista de claves en el mismo orden que las filas

En queries, Agent A carga estos archivos una sola vez, hace coseno vectorizado
en ~10ms, y recupera el registro completo de SHARD solo para el top-k.

Uso:
    python tools/build_embedding_cache.py --db data/spanish_bff_shard --dataset MMG/SpanishBFF
    python tools/build_embedding_cache.py --db data/shard_db --json data/demo_dataset.json

Dependencias adicionales (no incluidas en requirements.txt base):
    pip install datasets   # solo si usas --dataset (HuggingFace Hub)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_records_from_json(json_path: str) -> tuple:
    """Load (keys, texts) from a local JSON array file."""
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("El archivo JSON debe ser un array de objetos.")
    keys  = [str(r.get("lemma") or r.get("id") or r.get("term", f"rec_{i}")) for i, r in enumerate(records)]
    texts = [
        f"{r.get('lemma', '')}. {r.get('definition', r.get('text', ''))}"
        for r in records
    ]
    return keys, texts


def load_records_from_hub(dataset_name: str) -> tuple:
    """Load (keys, texts) from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "La librería 'datasets' es necesaria para cargar desde HuggingFace Hub.\n"
            "Instálala con: pip install datasets"
        ) from exc
    ds = load_dataset(dataset_name)["train"]
    keys  = [row["lemma"] for row in ds]
    texts = [f"{row['lemma']}. {row['definition']}" for row in ds]
    return keys, texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-computa embeddings para una base de datos SHARD."
    )
    parser.add_argument(
        "--db", required=True,
        help="Directorio de la base de datos SHARD donde se guardarán los archivos.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--json",
        help="Ruta a un archivo JSON local (array de objetos con 'lemma'/'definition').",
    )
    source.add_argument(
        "--dataset",
        help="Nombre del dataset en HuggingFace Hub (ej: MMG/SpanishBFF).",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Modelo sentence-transformers a usar (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Tamaño de batch para encoding (default: 512).",
    )
    args = parser.parse_args()

    db_dir    = Path(args.db)
    embed_file = db_dir / "embeddings.npy"
    keys_file  = db_dir / "embedding_keys.json"

    if not db_dir.exists():
        print(f"ERROR: El directorio de la DB no existe: {db_dir}", file=sys.stderr)
        sys.exit(1)

    # ── 1. Cargar registros ───────────────────────────────────────────────────
    if args.json:
        print(f"[1/3] Cargando registros desde JSON: {args.json}")
        keys, texts = load_records_from_json(args.json)
    else:
        print(f"[1/3] Cargando dataset desde HuggingFace Hub: {args.dataset}")
        keys, texts = load_records_from_hub(args.dataset)
    print(f"      {len(keys):,} registros.")

    # ── 2. Cargar modelo ──────────────────────────────────────────────────────
    print(f"[2/3] Cargando modelo: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device="cpu")

    # ── 3. Codificar y guardar ────────────────────────────────────────────────
    print(f"[3/3] Codificando {len(texts):,} registros en batches de {args.batch_size}...")
    all_embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"\nGuardando embeddings: {embed_file}")
    np.save(str(embed_file), all_embeddings.astype(np.float32))

    print(f"Guardando claves:     {keys_file}")
    with open(keys_file, "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False)

    size_mb = embed_file.stat().st_size / 1024 / 1024
    print(f"\nListo: {len(keys):,} registros, {size_mb:.1f} MB")
    print(f"Para usar en DASA:")
    print(f"  DASAConfig(use_shard_backend=True, shard_db_path={str(db_dir)!r})")


if __name__ == "__main__":
    main()
