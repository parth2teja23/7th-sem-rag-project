"""Build BM25 and dense (FAISS HNSW) indexes: indexes/<dataset>/{bm25, <encoder>[_<dim>]}.

Existing indexes are skipped unless --force (or unless their doc ids no longer match the
corpus, in which case they are rebuilt). Build times are merged into
results/indexes/build_times.json.

    python experiments/build_indexes.py --datasets scifact fiqa hotpot rfc --encoders bge-small gemma
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402


def _ids_match(meta_path: Path, corpus) -> bool:
    return set(json.loads(meta_path.read_text())["ids"]) == set(corpus)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(common.DATASETS))
    ap.add_argument("--encoders", nargs="+", default=list(common.ENCODER_NAMES))
    ap.add_argument("--dim", type=int, default=None, help="Matryoshka dim (gemma only)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from hrag.dense import DenseIndex
    from hrag.sparse import SparseIndex

    times_path = common.RESULTS_DIR / "indexes" / "build_times.json"
    times = json.loads(times_path.read_text()) if times_path.exists() else {}
    encoders: dict = {}

    for ds in args.datasets:
        if not common.dataset_available(ds):
            continue
        corpus = common.load_dataset(ds, common.SPLITS[ds][-1])[0]
        log(f"{ds}: {len(corpus)} docs")
        rec = times.setdefault(ds, {})

        bdir = common.index_dir(ds, "bm25")
        meta = bdir / "meta.json"
        if meta.exists() and not args.force and _ids_match(meta, corpus):
            print(f"skip {ds}/bm25 (exists)", flush=True)
        else:
            if meta.exists() and not args.force:
                log(f"{ds}/bm25 ids do not match corpus -> rebuilding")
            t0 = time.perf_counter()
            sp = SparseIndex()
            sp.build(corpus)
            sp.save(bdir)
            dt = time.perf_counter() - t0
            rec["bm25"] = {"seconds": round(dt, 2), "n_docs": len(corpus)}
            print(f"built {ds}/bm25 in {dt:.1f}s", flush=True)

        for name in args.encoders:
            key = common.encoder_key(name, args.dim)
            ddir = common.index_dir(ds, key)
            meta = ddir / "meta.json"
            if meta.exists() and not args.force and _ids_match(meta, corpus):
                print(f"skip {ds}/{key} (exists)", flush=True)
                continue
            if meta.exists() and not args.force:
                log(f"{ds}/{key} ids do not match corpus -> rebuilding")
            if key not in encoders:
                encoders[key] = common.make_encoder(name, args.dim)
            enc = encoders[key]
            log(f"encoding {ds} with {key} on {getattr(enc, 'device', '?')} ...")
            t0 = time.perf_counter()
            di = DenseIndex()
            di.build(corpus, enc, batch_size=args.batch_size)
            di.save(ddir)
            dt = time.perf_counter() - t0
            rec[key] = {"seconds": round(dt, 2), "n_docs": len(corpus),
                        "docs_per_s": round(len(corpus) / dt, 2) if dt > 0 else None,
                        "device": getattr(enc, "device", None), "batch_size": args.batch_size}
            print(f"built {ds}/{key} in {dt:.1f}s ({len(corpus) / max(dt, 1e-9):.1f} docs/s)",
                  flush=True)
            common.write_json(times_path, times)
        common.write_json(times_path, times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
