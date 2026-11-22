import numpy as np
import torch
from torch_geometric.data import Data
import os
import multiprocessing as mp
from rdkit import Chem

chunk_size = 200

def get_chunks(file_path, chunk_size):
    chunk = []

    with Chem.MultithreadedSDMolSupplier(file_path, numWriterThreads=4) as mols:
        for mol in mols:
            if mol is not None:
                chunk.append(mol)

            if chunk_size == len(chunk):
                yield chunk
                chunk = []

def process(chunk):
    chunk_to_save = []
    for mol in chunk:
        X = []
        edge_features = []

        encodings = {}
        encoded = None
        max_encode = 0

        edge_matrix_idx1 = []
        edge_matrix_idx2 = []
        adjacency_matrix_idx1 = []
        adjacency_matrix_idx2 = []

        if mol is None: return None

        for atom in mol.GetAtoms():
            for encoding, value in encodings.items():
                if value == atom.GetSymbol():
                    encoded = encoding
                else:
                    encoded = max_encode + 1
                    encoding.append(encoded, atom.GetSymbol())
                    max_encode += 1

            X.append([encoded, int(atom.GetHybridization()), atom.GetFormalCharge()])

        for bond in mol.GetBonds():
            edge_features.append([bond.GetBondType().real, float(bond.GetIsAromatic())])

            edge_matrix_idx1.append(bond.GetBeginAtomIdx())
            edge_matrix_idx2.append(bond.GetEndAtomIdx())

            edge_matrix_idx2.append(bond.GetBeginAtomIdx())
            edge_matrix_idx1.append(bond.GetEndAtomIdx())

        edge_matrix = torch.from_numpy(np.array([adjacency_matrix_idx1, adjacency_matrix_idx2]))
        X = torch.from_numpy(np.array(X))
        edge_features = torch.from_numpy(np.array(edge_features))
        graph = Data(x=X, edge_index=edge_matrix, edge_attr=edge_features)

        chunk_to_save.append(graph)
        if len(chunk_to_save) == chunk_size:
            if len(os.listdir('data/graph_chunks')) > 0:
                next_to_save = max([int(filename[:3]) + 1 for filename in os.listdir('data/graph_chunks')])
            else:
                next_to_save = 1
            torch.save(chunk_to_save, f'./data/graph_chunks/{next_to_save:04d}.pt')

if __name__ == '__main__':
    mols_path = 'data/BindingDB_All_2D.sdf'

    with mp.Pool(processes=5) as pool:
        results = pool.imap(process, get_chunks(mols_path, chunk_size))
        for _ in results: pass