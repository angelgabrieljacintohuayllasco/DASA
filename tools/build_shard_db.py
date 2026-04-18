"""
build_shard_db.py — Convierte un dataset JSON al formato SHARD binario.

Uso:
    python tools/build_shard_db.py
    python tools/build_shard_db.py --dataset data/demo_dataset.json --out data/shard_db

Qué hace:
1. Lee cada registro del JSON.
2. Escribe la clave (lemma/id) y el valor (JSON completo) en el shard correcto
   según FNV1a(clave) % num_shards → shard_NNN.bin
3. Construye el índice MinHash (index.minhash.bin) para búsqueda por similitud.

Para un dataset de 100 GB, este script se ejecuta UNA sola vez.
Después, Agent A consulta la base de datos en O(1) sin cargar nada en RAM.
"""

import argparse
import json
import sys
from pathlib import Path

# Asegura que SHARD sea importable desde cualquier directorio
_SHARD_ROOT = Path(__file__).parent.parent.parent / "SHARD"
if (_SHARD_ROOT / "shard" / "__init__.py").exists():
    sys.path.insert(0, str(_SHARD_ROOT))

try:
    from shard.storage.shard_writer import ShardWriter
    from shard.index.index_builder import IndexBuilder
except ImportError as e:
    print(f"ERROR: No se puede importar SHARD: {e}")
    print("Instala SHARD con:  pip install shard-db  (o: pip install -e /path/to/SHARD-main)")
    sys.exit(1)


def build(dataset_path: str, out_dir: str, num_shards: int = 16) -> None:
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Leyendo dataset: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"      {len(records)} registros encontrados.")

    print(f"[2/3] Escribiendo {num_shards} shards en: {out_dir}")
    with ShardWriter(str(out_dir), num_shards=num_shards) as writer:
        for record in records:
            # La clave es el lemma (o el id si no hay lemma)
            key = str(record.get("lemma") or record.get("id") or record.get("term", ""))
            if not key:
                continue
            value = json.dumps(record, ensure_ascii=False)
            writer.write(key, value)

    # Muestra qué shards se crearon
    shard_files = sorted(out_dir.glob("shard_*.bin"))
    print(f"      {len(shard_files)} archivos de shard creados.")
    for sf in shard_files:
        size_kb = sf.stat().st_size / 1024
        print(f"        {sf.name}  ({size_kb:.1f} KB)")

    print(f"[3/3] Construyendo índice MinHash de similitud...")
    builder = IndexBuilder(str(out_dir), num_shards=num_shards, num_hashes=64)
    for i, record in enumerate(records):
        key = str(record.get("lemma") or record.get("id") or record.get("term", ""))
        if not key:
            continue
        text = " ".join([
            str(record.get("lemma", "")),
            str(record.get("definition", "")),
            str(record.get("text", "")),
        ])
        builder.add(i, key, text)
    builder.build()
    print(f"      Índice guardado en: {out_dir / 'index.minhash.bin'}")

    print(f"\nBase de datos SHARD lista en: {out_dir}")
    print(f"\nPara usar en DASA:")
    print(f"  config = DASAConfig(")
    print(f"      use_shard_backend=True,")
    print(f"      shard_db_path={str(out_dir)!r},")
    print(f"      shard_num_shards={num_shards},")
    print(f"  )")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte JSON a base de datos SHARD.")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent.parent / "data" / "demo_dataset.json"),
        help="Ruta al archivo JSON de entrada.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "data" / "shard_db"),
        help="Directorio de salida para la base de datos SHARD.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=16,
        help="Número de shards (para 1 TB usar 1000+, para demo usar 16).",
    )
    args = parser.parse_args()
    build(args.dataset, args.out, args.num_shards)
