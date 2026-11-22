import numpy as np
import torch
from torch_geometric.data import Data
import os
import multiprocessing as mp
from rdkit import Chem

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

encodings = {'O': 1, 'C': 2, 'N': 3, 'Fe': 4, 'Co': 5, 'In': 6, 'Ag': 7, 'S': 8, 'Y': 9, 'Cu': 10, 'Ni': 11, 'La': 12, 'P': 13, 'Tb': 14, 'Br': 15, 'Mg': 16, 'Be': 17, 'Al': 18, 'Tm': 19, 'F': 20, 'Dy': 21, 'Cl': 22, 'Zn': 23, 'Pr': 24, 'Sm': 25, 'Er': 26, 'Sr': 27, 'Cd': 28, 'Th': 29, 'Am': 30, 'Mn': 31, 'Li': 32, 'Ce': 33, 'Lu': 34, 'Cs': 35, 'Ho': 36, 'Ca': 37, 'Gd': 38, 'Pb': 39, 'I': 40, 'Eu': 41, 'Tl': 42, 'Se': 43, 'Sn': 44, 'Cr': 45, 'Ba': 46, 'Yb': 47, 'Pu': 48, 'Pm': 49, 'Hg': 50, 'Nd': 51, 'Ge': 52, 'Sc': 53, 'Na': 54, 'Pd': 55, 'K': 56, 'Ga': 57, 'Cm': 58, 'V': 59, 'Bi': 60, 'Ti': 61, 'Np': 62, 'Zr': 63, 'Es': 64, 'Pa': 65, 'Ac': 66, 'Mo': 67, 'U': 68, 'W': 69, 'As': 70, 'Rb': 71, 'Hf': 72, 'Cf': 73, 'Pt': 74, 'Ra': 75, 'Bk': 76, 'Rh': 77, 'Ru': 78, 'Au': 79, 'Te': 80, 'No': 81, 'Fm': 82, 'Si': 83}

def process(chunk):
    chunk_to_save = []
    for mol in chunk:
        X = []
        edge_features = []

        edge_matrix_idx1 = []
        edge_matrix_idx2 = []

        for atom in mol.GetAtoms():
            X.append([encodings[atom.GetSymbol()], int(atom.GetHybridization()), atom.GetFormalCharge()])

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