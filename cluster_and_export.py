"""Cluster cached chains at <=60% sequence identity and export CA-only PDB files.

Usage:
    python cluster_and_export.py CACHE_DIR OUT_DIR [--id 0.6] [--cov 0.8] [--threads 8]

Reads ca_shard_*.h5 from build_ca_cache.py, collapses identical sequences (keeping the
chain with the most resolved CA), clusters the unique sequences with MMseqs2 and writes
one representative per cluster to OUT_DIR/structures/<SEQUENCE>.pdb. Sequences longer
than --max-name are named long_<sha1[:16]>.pdb instead (filesystem name limit); the
full mapping is in OUT_DIR/manifest.tsv (file, source chain, length, resolved, seq).
Requires the `mmseqs` binary on PATH (Linux/cluster).
"""
import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

THREE = dict(zip("ACDEFGHIKLMNPQRSTVWYX",
                 "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR UNK".split()))


def write_pdb(path, seq, ca):
    lines = []
    n = 0
    for i, (aa, xyz) in enumerate(zip(seq, ca), start=1):
        if np.isnan(xyz[0]):
            continue
        n += 1
        lines.append("ATOM  %5d  CA  %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C"
                     % (n % 100000, THREE.get(aa, "UNK"), i % 10000, *xyz))
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--id", type=float, default=0.6, dest="min_id")
    ap.add_argument("--cov", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-name", type=int, default=200)
    ap.add_argument("--no-cluster", action="store_true", help="skip MMseqs2 (testing only)")
    args = ap.parse_args()

    # 1. best chain per unique sequence
    best = {}  # seq -> (n_resolved, shard path, group name)
    for shard in sorted(Path(args.cache_dir).glob("ca_shard_*.h5")):
        with h5py.File(shard, "r") as h5:
            for name, g in h5.items():
                seq, nres = g.attrs["seq"], int(g.attrs["n_resolved"])
                if seq not in best or nres > best[seq][0]:
                    best[seq] = (nres, shard, name)
    print(f"{len(best)} unique sequences")

    # 2. cluster
    seqs = sorted(best)
    if args.no_cluster:
        reps = seqs
    else:
        with tempfile.TemporaryDirectory() as tmp:
            fa = Path(tmp) / "in.fasta"
            fa.write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(seqs)))
            subprocess.run(["mmseqs", "easy-cluster", str(fa), str(Path(tmp) / "out"), str(Path(tmp) / "tmp"),
                            "--min-seq-id", str(args.min_id), "-c", str(args.cov),
                            "--threads", str(args.threads)], check=True)
            tsv = (Path(tmp) / "out_cluster.tsv").read_text().splitlines()
        reps = [seqs[i] for i in sorted({int(l.split("\t")[0]) for l in tsv})]
    print(f"{len(reps)} cluster representatives at <={args.min_id:.0%} identity")

    # 3. export
    out = Path(args.out_dir)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.tsv", "w") as man:
        man.write("file\tsource_chain\tlength\tn_resolved\tseq\n")
        for seq in reps:
            nres, shard, name = best[seq]
            fname = seq if len(seq) <= args.max_name else "long_" + hashlib.sha1(seq.encode()).hexdigest()[:16]
            with h5py.File(shard, "r") as h5:
                write_pdb(out / "structures" / f"{fname}.pdb", seq, h5[name]["ca"][:])
            man.write(f"{fname}.pdb\t{name}\t{len(seq)}\t{nres}\t{seq}\n")
    print(f"wrote {len(reps)} PDB files to {out / 'structures'}")


if __name__ == "__main__":
    main()
