import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd

#Measuring distance between 2 CA (Fit in a structure too LOL):
def distance(): 

class MPNN(nn.Module): 
    # Message should have the same shape of the original input. Edge should be bidirection (2 vectors per A-B relations. A->B. B->A)
    # Edge is often stored as a list (sender, receiver) index pairs. Ex: With 3 residues long peptide, we have edge 0 = (0,1), edge 1 = (1,0), etc... 4 totals message being sent
    # Edge shape: [4, 2] (distance + order in sequence)  +  [4,4] (residue identity for receiver) + [4,4] (resideu idenitity for sender) = [4,10]

    def __init__(self, d_edge, two_d):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(two_d + d_edge, two_d/2), nn.ReLU(), nn.Linear(two_d/2,two_d/2)) #Adding a matrix after the ReLU allows for information capture being created here.
        self.update = nn.Sequential(nn.Linear(two_d, 4), nn.ReLU(), nn.Linear(4, 4))
        

    #X: Shape [n, D]
    #edge: shape [4, 2] (General shape: [# interaction, dimensions of info to convey] or shortly [e, D_e])
    #index_mapping: [#senders, #receivers] or [4,4] for this case
    #Matrix index goes with first row and then follows down. Ex: h[receivers] (receivers should be a list of numbers in this case. Not a matrix with more than 1 dimension)
    def forward(self, h, edge, index_mapping):
        index_mapping = torch.flatten(index_mapping)
        assert index_mapping.dim() == 1, f"Wrong shape when flattening index_map {index_mapping.dim}"
        senders = h[index_mapping[::2]]
        receivers = h[index_mapping[1::2]]
        merged_e_index = torch.concatenate((edge, senders, receivers), dim=1)
        
        message = self.message(merged_e_index)
        update = torch.zeros(h.shape).index_add(0, index_mapping[1::2], message)

        update = self.update(torch.concatenate((h, update), dim=1))
        return update

#Testing trial

#N, d, E, D_e = 3, 4, 4, 2
#Creating Edge Index: 
def attention(k, seq_len):
    edge_index = []
    for res in range(seq_len):
        for i in range(1,k+1):
            if res + i < seq_len:
                edge_index.append([res, res + i])
                edge_index.append([res+i, res])

    return torch.tensor(edge_index)

#edge_index = attention(1, 3)

#h = torch.randn(N, d)
#e = torch.randn(E, D_e)

#Layer1 = MPNN(2*d+D_e, d*2)
#out = Layer1(h, e, edge_index)

#Check output shape
#assert out.shape == h.shape, f"Mismatch shape between output {out.shape} and input {h.shape}"

import esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
batch_converter = alphabet.get_batch_converter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('mnt/Research/Protores/amp_regime_proteolysis_training_set_v2.csv')
k = 5 
df['edge_index'] = df['aa_seq'].apply(lambda seq: attention(k, len(seq)))
 

class PeptideGraph(Dataset): 
    def __init__(self, batch_size, numbers_res_to_attend_to): 
        self.real_embeddings = []
        for start in range(0, len(df['aa_seq']), batch_size):
            seqs = df['aa_seq'].iloc[start: start + batch_size].tolist()
            data = [(idx, seq) for idx, seq in enumerate(seqs)]

            with torch.no_grad():
                _, _, tokens = batch_converter(data) #Remember that batch_converter requires (idx, seq) besides the sequence identity.

                output = model(tokens.to(device), repr_layers=10)
                emb = output['representation'][33] # The shape for output is [batch, Lmax*2, 1280]

            self.real_embeddings.append([emb[idx, 1:len(seq)+1].cpu() for idx, seq in enumerate(seqs)]) 

        self.read_embeddings = torch.tensor(self.real_embeddings) #Expected shape [L, 1280]
        assert self.real_embeddings.dim() == 3, "Wrong shape for real_embeddings"
        self.edge_index = [attention(numbers_res_to_attend_to, len(seq)) for seq in df['aa_seq'].tolist()]
        df['index_relative'] = #Encode relative distance on the chain here. 
        
        self.MPNN = MPNN(2, 2560)
        






        