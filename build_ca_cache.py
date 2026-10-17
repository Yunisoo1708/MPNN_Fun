"""Build a cache of protein-only PDB chains: sequence -> CA coordinates.

Usage:
    python build_ca_cache.py MMCIF_DIR OUT_DIR [--shard I N] [--min-len 10]

MMCIF_DIR is searched recursively for *.cif / *.cif.gz. Each shard writes
OUT_DIR/ca_shard_{I:04d}.h5 with one group per chain, named <pdbid>_<authchain>:
    seq  (attr, str)        full SEQRES sequence (one-letter, X = unknown)
    ca   (L, 3 float32)     CA coordinates, NaN where unresolved
Chains whose residue types disagree with SEQRES at any resolved position are dropped.
Entries containing any non-protein polymer (DNA/RNA/D-peptide/etc.) are skipped
entirely, so only protein-alone structures are cached. Slurm array usage:
    python build_ca_cache.py pdb_mmcif cache --shard $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
"""
import argparse
import sys
from pathlib import Path

import gemmi
import h5py
import numpy as np


def one_letter(name):
    info = gemmi.find_tabulated_residue(name)
    return info.one_letter_code.upper() if info and info.one_letter_code.strip() else "X"


def process_entry(path, min_len):
    """Return (pdbid, [(chain_id, seq, coords)], skip_reason)."""
    st = gemmi.read_structure(str(path))
    pdbid = Path(path).name.split(".")[0].lower()
    st.setup_entities()
    polymers = [e for e in st.entities if e.entity_type == gemmi.EntityType.Polymer]
    if not polymers:
        return pdbid, [], "no polymer"
    if any(e.polymer_type != gemmi.PolymerType.PeptideL for e in polymers):
        return pdbid, [], "non-protein polymer"

    out = []
    for chain in st[0]:
        poly = chain.get_polymer()
        if len(poly) == 0:
            continue
        ent = st.get_entity_of(poly)
        if ent is None or not ent.full_sequence:
            continue
        seq = "".join(one_letter(n) for n in ent.full_sequence)
        if len(seq) < min_len:
            continue
        coords = np.full((len(seq), 3), np.nan, dtype=np.float32)
        bad = False
        for res in poly:
            i = res.label_seq
            if i is None or i < 1 or i > len(seq):
                bad = True
                break
            if one_letter(res.name) != seq[i - 1]:
                bad = True  # sequence-structure mismatch: drop the whole chain
                break
            atom = res.find_atom("CA", "*")
            if atom is not None:
                p = atom.pos
                coords[i - 1] = (p.x, p.y, p.z)
        if bad or np.isnan(coords[:, 0]).all():
            continue
        out.append((chain.name, seq, coords))
    return pdbid, out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mmcif_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--shard", nargs=2, type=int, default=[0, 1], metavar=("I", "N"))
    ap.add_argument("--min-len", type=int, default=10)
    args = ap.parse_args()

    i, n = args.shard
    root = Path(args.mmcif_dir)
    files = sorted(list(root.rglob("*.cif.gz")) + list(root.rglob("*.cif")))[i::n]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ca_shard_{i:04d}.h5"

    n_chains = n_skipped = n_failed = 0
    with h5py.File(out_path, "w") as h5:
        for f in files:
            try:
                pdbid, chains, reason = process_entry(f, args.min_len)
            except Exception as e:
                n_failed += 1
                print(f"FAIL {f.name}: {e}", file=sys.stderr)
                continue
            if reason:
                n_skipped += 1
                continue
            for cid, seq, coords in chains:
                g = h5.create_group(f"{pdbid}_{cid}")
                g.attrs["seq"] = seq
                g.attrs["n_resolved"] = int((~np.isnan(coords[:, 0])).sum())
                g.create_dataset("ca", data=coords, compression="gzip")
                n_chains += 1
    print(f"shard {i}/{n}: {len(files)} files, {n_chains} chains, "
          f"{n_skipped} entries skipped (non-protein), {n_failed} failed -> {out_path}")


if __name__ == "__main__":
    main()
