from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import AllChem, MACCSkeys


def molecule_to_maccs(x):
    maccs_ls = []
    maccs = MACCSkeys.GenMACCSKeys(x).ToBitString()
    # maccs_ls.append(int(maccs[bit]) for bit in range(len(maccs)))
    for bit in range(len(maccs)):
        maccs_ls.append(int(maccs[bit]))
    return maccs_ls
    # return MACCSkeys.GenMACCSKeys(x).ToBitString()


def molecule_to_fcfp4(x):
    fcfp4_ls = []
    fcfp4 = AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048, useFeatures=True).ToBitString()
    # fcfp4_ls.append(int(bit) for bit in fcfp4)
    for bit in range(len(fcfp4)):
        fcfp4_ls.append(int(fcfp4[bit]))
    return fcfp4_ls
    # return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048, useFeatures=True).ToBitString()


def molecule_to_fingerprint(smiles_ls):
    data = pd.DataFrame()
    lost = pd.DataFrame()
    maccs = pd.DataFrame(columns=range(167))
    fcfp4 = pd.DataFrame(columns=range(2048))
    length, length_ = 0, 0

    for i in tqdm(range(len(smiles_ls))):
        mol = AllChem.MolFromSmiles(smiles_ls[i])
        if mol is not None:
            data.at[length, 'smiles'] = smiles_ls[i]
            maccs.loc[length] = molecule_to_maccs(mol)
            fcfp4.loc[length] = molecule_to_fcfp4(mol)
            length += 1
        else:
            # print(f"{i}:something wrong")
            lost.at[length_, 'smiles'] = smiles_ls[i]
            length_ += 1

    data.to_csv("../dataset/Hepatotoxicity/Ai/test/mito_smiles.csv", index=False)
    maccs.to_csv("../dataset/Hepatotoxicity/Ai/test/mito_maccs.csv", index=False)
    fcfp4.to_csv("../dataset/Hepatotoxicity/Ai/test/mito_fcfp4.csv", index=False)

    lost.to_csv("../dataset/Hepatotoxicity/Ai/test/mito_lost.csv", index=False)
    return f'success:{len(data)} wrong:{len(lost)}'
