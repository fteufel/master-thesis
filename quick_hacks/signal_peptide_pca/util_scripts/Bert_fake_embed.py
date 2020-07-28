import numpy as np
import torch
import pandas as pd
from tape import TAPETokenizer, ProteinBertModel
import pickle

tokenizer = TAPETokenizer(vocab='iupac')
model = ProteinBertModel.from_pretrained('bert-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def unirep_encode(series):
    with torch.no_grad():
        series = list(series)
        output = []
        for i in range(len(series)):
            print((i, len(series[i])))
            x = series[i]
            l = list(x)
            random.shuffle(l)
            x = ''.join(l)
            x = tokenizer.encode(x)
            output.append(model(torch.tensor(x).unsqueeze(0).to(device))[0].mean(axis =1).detach().cpu().numpy())
        series = np.concatenate(output, axis =0)

        return series

def encode_and_dump(series, dumpname):
    series = unirep_encode(series)
    pickle.dump( series, open( f"{dumpname}.pkl", "wb" ) )


print('reading table')
df = pd.read_csv('preprocessed_dump.tsv', sep ='\t')

print('embedding...')
savedict = {}
encode_and_dump(df.loc[df['Taxonomic lineage (GENUS)'] == 'Plasmodium', 'peptide'], 'pla_pep_shuffled')


encode_and_dump(df.loc[df['Taxonomic lineage (GENUS)'] == 'Plasmodium', 'protein'], 'pla_pro_shuffled')
encode_and_dump(df.loc[df['Taxonomic lineage (GENUS)'] != 'Plasmodium', 'protein'], 'euk_pro_shuffled')

encode_and_dump(df.loc[df['Taxonomic lineage (GENUS)'] != 'Plasmodium', 'peptide'], 'euk_pep_shuffled')
