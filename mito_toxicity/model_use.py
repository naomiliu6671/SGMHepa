import os.path

import joblib
import pandas as pd

from mito_toxicity.to_fingerprint import molecule_to_fingerprint

def use_model(run_data):
    smiles_path = '../dataset/Hepatotoxicity/Ai/test/mito_smiles.csv'
    maccs_path = '../dataset/Hepatotoxicity/Ai/test/mito_maccs.csv'
    fcfp4_path = '../dataset/Hepatotoxicity/Ai/test/mito_fcfp4.csv'
    if os.path.exists(smiles_path) == False:
        message = molecule_to_fingerprint(run_data)
        print(message)

    smiles = pd.read_csv(smiles_path)
    maccs = pd.read_csv(maccs_path)
    fcfp4 = pd.read_csv(fcfp4_path)

    data = pd.concat([maccs, fcfp4], axis=1)
    data.columns = [i for i in range(167+2048)]

    model = joblib.load('../mito_toxicity/mito_toxicity_model.joblib')

    tox = model.predict(data)
    mito_tox = smiles
    mito_tox['mito_tox'] = tox
    mito_tox.to_csv('../dataset/Hepatotoxicity/Ai/test/mito_tox_pred.csv', index=False)
    return mito_tox
