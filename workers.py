import numpy as np
import torch
from torch_geometric.data import Data
import os
import multiprocessing as mp
from rdkit import Chem
from joblib import load

chunk_size = 200

def get_chunks(file_path, chunk_size):
    chunk = []
    supplier = Chem.SmilesMolSupplier(file_path, delimiter=',', smilesColumn=0, nameColumn=-1, titleLine=True)

    for mol in supplier:
        if mol is None: continue
        chunk.append(mol)

        if len(chunk) == chunk_size:
            yield chunk
            chunk = []

encodings = load('./data/one-hot.joblib')

def process(chunk):
    chunk_to_save = []
    for mol in chunk:
        X = []
        edge_features = []

        edge_matrix_idx1 = []
        edge_matrix_idx2 = []

        for atom in mol.GetAtoms():
            X.append([*encodings[atom.GetSymbol()], int(atom.GetHybridization()), atom.GetFormalCharge()])

        for bond in mol.GetBonds():
            features_for_edge = [bond.GetBondType().real, float(bond.GetIsAromatic())]
            edge_features.append(features_for_edge)
            edge_features.append(features_for_edge)

            edge_matrix_idx1.append(bond.GetBeginAtomIdx())
            edge_matrix_idx2.append(bond.GetEndAtomIdx())

            edge_matrix_idx2.append(bond.GetBeginAtomIdx())
            edge_matrix_idx1.append(bond.GetEndAtomIdx())

        edge_matrix = torch.from_numpy(np.array([edge_matrix_idx1, edge_matrix_idx2]))
        X = torch.from_numpy(np.array(X))
        edge_features = torch.from_numpy(np.array(edge_features))
        graph = Data(x=X, edge_index=edge_matrix, edge_attr=edge_features)

        chunk_to_save.append(graph)
        if len(chunk_to_save) == chunk_size:
            if len(os.listdir('data/graph_chunks')) > 0:
                next_to_save = max([int(filename[:4]) + 1 for filename in os.listdir('data/graph_chunks')])
            else:
                next_to_save = 1
            torch.save(chunk_to_save, f'./data/graph_chunks/{next_to_save:04d}.pt')

if __name__ == '__main__':
    mols_path = 'data/train_input.csv'

    with mp.Pool(processes=5) as pool:
        results = pool.imap(process, get_chunks(mols_path, chunk_size))
        for _ in results: pass