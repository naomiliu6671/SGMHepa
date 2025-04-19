import pickle

import numpy as np
import torch
from rdkit.Chem import AllChem
from rdkit import Chem
from collections import defaultdict

from torch import tensor


def get_atom_features(mol, symbol2idx, use_chirality=False):
    atom_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        degree = atom.GetTotalDegree()
        formal_charge = atom.GetFormalCharge()
        num_hydrogens = atom.GetTotalNumHs()
        hybridization = atom.GetHybridization().real
        aromaticity = atom.GetIsAromatic()
        chirality = atom.GetChiralTag().real if use_chirality else 0
        atom_features.append([
            symbol,
            degree,
            formal_charge,
            num_hydrogens,
            hybridization,
            int(aromaticity),
            chirality
        ])

    atom_features_encoded = []
    for feat in atom_features:
        feat[0] = symbol2idx[feat[0]]
        atom_features_encoded.append(feat)
    x = torch.tensor(atom_features_encoded, dtype=torch.float32)
    return x, symbol2idx


def load_smiles_data(args, test_data=''):
    data_file = f"dataset/{args.property_name}/{args.property_name}_{args.property_add}_train2.csv"
    datafile_ls = [data_file]
    if test_data == '':
        data_file = f"dataset/{args.property_name}/{args.property_name}_{args.property_add}_test2.csv"
    else:
        data_file = test_data
    datafile_ls.append(data_file)
    max_len = 256

    node_types = set()
    node_types.add("<mask>")
    # node_types.add("<pad>")
    # node_types.add("<unk>")
    label_types = set()
    mito_type = set()
    tr_te_len = []

    for data_file in datafile_ls:
        file = open(data_file, "r")
        data_len = 0
        for line in file:
            if data_len == 0:
                data_len += 1
                continue
            smiles = line.split(",")[1]
            s = smiles.split(' ')
            mol = smiles.replace(' ', '')
            if smiles is None:
                continue
            for atom in mol:
                s.append(atom)
            node_types |= set(s)

            label = line.strip().split(',')[2]
            mito = line.strip().split(',')[3]
            label_types.add(label)
            mito_type.add(mito)
            data_len += 1
        file.close()
        tr_te_len.append(data_len - 1)

    node2index = {n: i for i, n in enumerate(node_types)}
    label2index = {l: i for i, l in enumerate(label_types)}
    mito2index = {m: i for i, m in enumerate(mito_type)}

    train_edges = []
    train_atom_features = []
    # train_features = []
    train_sequence = []
    train_labels = torch.zeros(tr_te_len[0])
    train_mitos = torch.zeros(tr_te_len[0])
    test_edges = []
    test_atom_features = []
    # test_features = []
    test_sequence = []
    test_labels = np.zeros(tr_te_len[1])
    test_mitos = np.zeros(tr_te_len[1])
    test_smiles = []
    symbol2idx = defaultdict(lambda: len(symbol2idx))

    for data_i in range(2):
        data_file = datafile_ls[data_i]
        file = open(data_file, "r")
        flag = True
        for line in file:
            if flag:
                flag = False
                continue
            if len(line.strip().split(',')) < 3:
                continue
            smiles = line.split(",")[1]
            label = line.strip().split(',')[2]
            mito = line.strip().split(',')[3]
            mol_str = smiles.replace(' ', '')
            mol = AllChem.MolFromSmiles(smiles.replace(' ', ''))
            if mol is None:
                continue

            # feature = torch.zeros(len(mol.GetAtoms()), len(node_types))
            l = 0
            smiles_seq = []
            for atom in mol_str:
                smiles_seq.append(node2index[atom])
                l += 1
            edges = []
            atom_featrues, symbol2idx = get_atom_features(mol, symbol2idx, use_chirality=True)
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                typ = bond.GetBondType()
                edges.extend([[i, j], [j, i]])
                if typ == Chem.rdchem.BondType.DOUBLE:
                    edges.extend([[i, j], [j, i]])
                elif typ == Chem.rdchem.BondType.TRIPLE:
                    edges.extend([[i, j], [j, i]])
            if data_i == 0:
                train_labels[len(train_sequence)] = int(label2index[label])
                train_mitos[len(train_sequence)] = int(mito2index[mito])
                train_atom_features.append(atom_featrues)
                if edges:
                    train_edges.append(torch.tensor(edges, dtype=torch.long).t().contiguous())
                else:
                    train_edges.append(torch.tensor([], dtype=torch.long).view(2,0))
                # train_features.append(torch.FloatTensor(feature).to(args.device))
                train_sequence.append(torch.tensor(smiles_seq))
            else:
                test_smiles.append(smiles)
                test_labels[len(test_sequence)] = int(label2index[label])
                test_mitos[len(test_sequence)] = int(mito2index[mito])
                test_atom_features.append(atom_featrues)
                test_edges.append(torch.tensor(edges, dtype=torch.long).t().contiguous())
                # test_features.append(torch.FloatTensor(feature).to(args.device))
                test_sequence.append(torch.tensor(smiles_seq))
        file.close()
    train_data = {}
    train_data['atom_features'] = train_atom_features
    train_data['edges'] = train_edges
    # train_data['features'] = train_features
    train_data['sequence'] = train_sequence
    train_data['dict'] = node2index

    test_data = {}
    test_data['atom_features'] = test_atom_features
    test_data['edges'] = test_edges
    # test_data['features'] = test_features
    test_data['sequence'] = test_sequence
    test_data['smiles'] = test_smiles

    return train_data, train_labels, train_mitos, test_data, test_labels, test_mitos, node2index, node_types, len(symbol2idx)+1


def delete_mol(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    dele_data = tensor([788, 3184, 2380, 2057,  505])
    for i in range(len(data['train_data']['sequence'])):
        if data['train_data']['sequence'][i].shape == dele_data.shape:
            if torch.equal(data['train_data']['sequence'][i], dele_data):
                data['train_data']['sequence'].pop(i)
                data['train_data']['atom_features'].pop(i)
                data['train_data']['edges'].pop(i)
                print(f'sucessfully delete {i}')
                pickle.dump(data, open(data_path, "wb"))


# delete_mol('/data/yliu/hepa_mito_model/dataset/Hepatotoxicity/Hepatotoxicity_mito_remove_pre2.pkl')
