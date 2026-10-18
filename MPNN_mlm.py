"""BERT-style masked-residue pretraining: an MPNN over the 3D neighbor graph predicts the hidden amino acids."""
import os
from io import StringIO
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser
from torch.utils.data import DataLoader, Dataset, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TSV_PATH = '/mnt/c/raw_protein_structure/clusters/manifest.tsv'
CSV_PATH = '/mnt/c/RFdiffusion/amp_regime_proteolysis_training_set_v2.csv'
SEQ_COL = 'aa_seq'
CACHE = 'ca_cache.pt'  # {sequence: CA coords}, so each peptide is folded only once

K_NN = 8        # each residue receives messages from its K nearest residues in 3D
D = 128         # hidden size
N_LAYERS = 3
N_RBF = 16      # Gaussian bins used to encode distance
MASK_P = 0.15

AA = 'ACDEFGHIKLMNPQRSTVWY'
N_AA = len(AA)
MASK_ID, UNK_ID = N_AA, N_AA + 1  # vocab = 20 amino acids + [MASK] + unknown
aa_idx = {a: i for i, a in enumerate(AA)}


# ---------- Structure (ESMFold, cached) ----------
_esmfold = None


def fold_ca(seq):
    global _esmfold
    if _esmfold is None:
        import esm
        _esmfold = esm.pretrained.esmfold_v1().eval().to(device)
    with torch.no_grad():
        pdb = _esmfold.infer_pdb(seq)
    chain = next(PDBParser(QUIET=True).get_structure('pep', StringIO(pdb)).get_chains())
    ca = torch.tensor(np.stack([r['CA'].coord for r in chain if 'CA' in r]))  # [L, 3]
    assert len(ca) == len(seq), f'CA count {len(ca)} != length {len(seq)} for {seq}'
    return ca


def load_coords(seqs):
    cache = torch.load(CACHE) if os.path.exists(CACHE) else {}
    for s in seqs:
        if s not in cache:
            cache[s] = fold_ca(s)
    torch.save(cache, CACHE)
    return [cache[s] for s in seqs]

#---Building Cache for case that structures are already built ---------
def read_pdb(path): 
    structure = PDBParser(QUIET=True).get_structure('pep', path)
    model = structure[0]
    chain = next(model.get_chains())
    return torch.tensor(np.stack([res['CA'].coord for res in chain if 'CA' in res]))

def _read_ca(job):
    """Pool worker: (path, seq) -> (seq, float32 [L,3] ndarray), or None if unreadable / length mismatch."""
    path, seq = job
    try:
        chain = next(PDBParser(QUIET=True).get_structure('pep', path)[0].get_chains())
        ca = np.stack([res['CA'].coord for res in chain if 'CA' in res])
    except Exception:
        return None
    return (seq, ca) if len(ca) == len(seq) else None


def build_cache(pdb_dir, dataset_path=TSV_PATH, workers=16):
    Dataset = pd.read_csv(dataset_path, sep="\t")
    jobs = [(os.path.join(pdb_dir, f), s) for f, s in zip(Dataset['file'], Dataset['seq'])]
    Cache = torch.load(CACHE) if os.path.exists(CACHE) else {}  # keep earlier (ESMFold) entries; PDB entries override same sequences
    print(f'build_cache: starting from {len(Cache)} cached sequences', flush=True)
    with Pool(workers) as pool:  # file reads on /mnt/c are latency-bound, so many workers help a lot
        for i, res in enumerate(pool.imap(_read_ca, jobs, chunksize=64), 1):
            if res is not None:
                Cache[res[0]] = torch.from_numpy(res[1])
            if i % 10000 == 0:
                print(f'build_cache: {i}/{len(jobs)} files read, {len(Cache)} kept', flush=True)
    print(f'build_cache: done, {len(Cache)} of {len(jobs)} chains kept', flush=True)
    torch.save(Cache, CACHE)
    return Cache



# ---------- Graph ----------
def knn_graph(ca, k=K_NN):
    """Edges from each residue's k nearest 3D neighbors. Returns edge_index [E, 2] (sender, receiver) and edge_attr [E, 2] (distance, signed seq offset)."""
    L = len(ca)
    k = min(k, L - 1)
    dist = torch.cdist(ca, ca)
    dist.fill_diagonal_(float('inf'))
    send = dist.topk(k, largest=False).indices              # [L, k]
    recv = torch.arange(L).unsqueeze(1).expand(-1, k)       # [L, k]
    ei = torch.stack((send.reshape(-1), recv.reshape(-1)), dim=1)
    d = dist[ei[:, 0], ei[:, 1]]
    offset = (ei[:, 0] - ei[:, 1]).float()
    return ei, torch.stack((d, offset), dim=1)


class PeptideStructures(Dataset):
    def __init__(self, seqs):
        self.items = []
        for seq, ca in zip(seqs, load_coords(seqs)):
            tokens = torch.tensor([aa_idx.get(a, UNK_ID) for a in seq])
            ei, ea = knn_graph(ca)
            self.items.append((tokens, ei, ea))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(items):
    tokens, eis, eas, offset = [], [], [], 0
    for t, ei, ea in items:
        tokens.append(t)
        eis.append(ei + offset)
        eas.append(ea)
        offset += len(t)
    return torch.cat(tokens), torch.cat(eis), torch.cat(eas)


# ---------- Model ----------
class RBF(nn.Module):
    def __init__(self, n=N_RBF, dmax=20.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0, dmax, n))
        self.width = dmax / n

    def forward(self, d):
        return torch.exp(-(((d.unsqueeze(-1) - self.centers) / self.width) ** 2))


class MPNNLayer(nn.Module):
    def __init__(self, d, d_edge):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * d + d_edge, 4*d), nn.ReLU(), nn.Linear(4*d, 2*d), nn.ReLU(), nn.Linear(2*d, d))
        self.update = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, edge_feat, edge_index):
        send, recv = edge_index[:, 0], edge_index[:, 1]
        msg = self.message(torch.cat((edge_feat, h[send], h[recv]), dim=1))
        deg = torch.bincount(recv, minlength=len(h)).clamp(min=1).unsqueeze(1)
        agg = torch.zeros_like(h).index_add(0, recv, msg) / deg
        return self.norm(h + self.update(torch.cat((h, agg), dim=1)))  # residual + norm


class MaskedMPNN(nn.Module):
    def __init__(self, d=D, n_layers=N_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(N_AA + 2, d)  # learned residue embeddings (NOT ESM: those would leak the masked identity)
        self.rbf = RBF()
        self.layers = nn.ModuleList(MPNNLayer(d, N_RBF + 1) for _ in range(n_layers))
        self.head = nn.Linear(d, N_AA)

    def encode(self, tokens, edge_index, edge_attr):
        h = self.embed(tokens)
        offset = edge_attr[:, 1].clamp(-8, 8) / 8 #Put the offset on range of -1 to 1. A large number early can bias the training weights easy. 
        edge_feat = torch.cat((self.rbf(edge_attr[:, 0]), offset.unsqueeze(1)), dim=1)
        for layer in self.layers:
            h = layer(h, edge_feat, edge_index)
        return h  # [N, d]; reuse this for downstream scoring

    def forward(self, tokens, edge_index, edge_attr):
        return self.head(self.encode(tokens, edge_index, edge_attr))  # [N, 20] logits


# ---------- BERT masking ----------
def mask_tokens(tokens):
    """Pick 15% of residues; of those 80% -> [MASK], 10% -> random residue, 10% unchanged. Unpicked labels are -100 (ignored)."""
    sel = (torch.rand(len(tokens), device=tokens.device) < MASK_P) & (tokens < N_AA)
    labels = torch.where(sel, tokens, torch.full_like(tokens, -100))
    r = torch.rand(len(tokens), device=tokens.device)
    inp = tokens.clone()
    inp[sel & (r < 0.8)] = MASK_ID
    rand_sel = sel & (r >= 0.8) & (r < 0.9)
    inp[rand_sel] = torch.randint(0, N_AA, (int(rand_sel.sum()),), device=tokens.device)
    return inp, labels


def run_epoch(net, loader, opt=None):
    net.train(opt is not None)
    total_loss, correct, count = 0.0, 0, 0
    for tokens, ei, ea in loader:
        tokens, ei, ea = tokens.to(device), ei.to(device), ea.to(device)
        inp, labels = mask_tokens(tokens)
        keep = labels != -100
        if not keep.any():
            continue
        with torch.set_grad_enabled(opt is not None):
            logits = net(inp, ei, ea)
            loss = F.cross_entropy(logits[keep], labels[keep])
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        n = int(keep.sum())
        total_loss += loss.item() * n
        correct += int((logits[keep].argmax(-1) == labels[keep]).sum())
        count += n
    return total_loss / count, correct / count


if __name__ == '__main__':
    seqs = pd.read_csv(CSV_PATH)[SEQ_COL].tolist()
    build_cache('/mnt/c/raw_protein_structure/clusters/structures', TSV_PATH)
    data = PeptideStructures(seqs)
    n_val = max(1, len(data) // 10)
    train_set, val_set = random_split(data, [len(data) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=32, collate_fn=collate)

    net = MaskedMPNN().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(100):
        tr_loss, tr_acc = run_epoch(net, train_loader, opt)
        va_loss, va_acc = run_epoch(net, val_loader)
        print(f'epoch {epoch}: train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}')

    torch.save(net.state_dict(), 'mpnn_mlm_2.pt')
