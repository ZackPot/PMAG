import numpy as np
import torch
from torch_geometric.data import Data
import os
import multiprocessing as mp
from rdkit import Chem
from joblib import load

chunk_size = 200
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

def get_chunks(file_path, chunk_size):
    index = 0
    chunk = []
    supplier = Chem.SmilesMolSupplier(file_path, delimiter=',', smilesColumn=0, nameColumn=-1, titleLine=True)

    for mol in supplier:
        if mol is None: continue
        chunk.append(mol)

        if len(chunk) == chunk_size:
            yield (chunk, index)
            chunk = []
            index += 1

    if chunk: yield chunk, index

encodings = load('./data/one-hot.joblib')

def process(chunk):
    chunk_to_save = []
    lgk1_list = []

    for mol in chunk[0]:
        lgk1 = float(mol.GetProp('Experimental LgK1'))
        lgk1_list.append(lgk1)

    lgk1_list = np.array(lgk1_list)
    lgk1_mean = np.mean(lgk1_list)
    lgk1_std = np.std(lgk1_list)

    for mol in chunk[0]:
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

        lgk1 = torch.tensor(float(mol.GetProp('Experimental LgK1')))
        norm_lgk1 = (lgk1 - lgk1_mean) / lgk1_std

        graph = Data(x=X, edge_index=edge_matrix, edge_attr=edge_features, y=norm_lgk1)

        chunk_to_save.append(graph)

    if len(chunk_to_save) == chunk_size:
        torch.save(chunk_to_save, f'./data/graph_chunks/{chunk[1]:04d}.pt')

if __name__ == '__main__':
    mols_path = 'data/train_input.csv'

    with mp.Pool(processes=5) as pool:
        results = pool.imap(process, get_chunks(mols_path, chunk_size))
        for _ in results: pass