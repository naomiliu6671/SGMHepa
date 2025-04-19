import pandas as pd
from tqdm import tqdm

from parsers import get_args
from mito_toxicity.model_use import use_model

args = get_args()

dataset_name = 'Hepatotoxicity'

mito_toxicity = pd.read_csv(f'{args.root_dir}/dataset/{dataset_name}/mito_toxicity.csv')
hepatotoxicity = pd.read_csv(f'{args.root_dir}/dataset/{dataset_name}/Ai_test.csv')

mito_toxicity.columns = ['smiles', 'mito_toxicity']
hepatotoxicity.columns = ['smiles', 'hepatotoxicity']

data = pd.merge(hepatotoxicity, mito_toxicity, on='smiles', how='left')
no_mito_smiles = []
have_mito_smiles = pd.DataFrame(columns=['smiles', 'hepatotoxicity', 'mito_toxicity'])

for i in range(len(data)):
    if pd.isnull(data['mito_toxicity'].iloc[i]):
        no_mito_smiles.append(data['smiles'].iloc[i])
    else:
        have_mito_smiles.loc[len(have_mito_smiles)] = data.loc[i]
have_mito_smiles.to_csv(f'{args.root_dir}/dataset/{dataset_name}/Ai/test/have_mito.csv', index=False)
print(f'Have mito_toxicity: {len(have_mito_smiles)}')

pred_mito_tox = use_model(no_mito_smiles)
print(f'Predicted {len(pred_mito_tox)} mito_toxicity')
if len(pred_mito_tox) == len(no_mito_smiles):
    for i in range(len(pred_mito_tox)):
        smi, mito = pred_mito_tox.loc[i]
        data.loc[data['smiles'] == smi, 'mito_toxicity'] = mito
    data.to_csv(f'{args.root_dir}/dataset/Hepatotoxicity/Hepatotoxicity_mito_Ai_test.csv', index=False)
else:
    print('Error: The length of predicted mito_toxicity is not equal to the length of no_mito_smiles')
