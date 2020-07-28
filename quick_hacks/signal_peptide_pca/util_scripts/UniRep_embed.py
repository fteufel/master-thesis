import numpy as np
import torch
import pandas as pd
from tape import TAPETokenizer, UniRepModel
import pickle

tokenizer = TAPETokenizer(vocab='unirep')
model = UniRepModel.from_pretrained('babbler-1900')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def unirep_encode(series):
    with torch.no_grad():
        series = series.apply(lambda x: tokenizer.encode(x))
        series = list(series)
        #series = series.apply(lambda x: model(torch.tensor(x).unsqueeze(0).to(device))[0].mean(axis =1).detach().cpu().numpy())
        output = []
        for i in range(len(series)):
            print((i, len(series[i])))
            x = series[i]
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
encode_and_dump(df.loc[df['Taxonomic lineage (GENUS)'] == 'Plasmodium', 'peptide'], 'pla_pep')


encode_and_dump(df.loc[df['Taxonomic lineage (GENUS)'] == 'Plasmodium', 'protein'], 'pla_pro')
encode_and_dump(df.loc[df['Taxonomic lineage (GENUS)'] != 'Plasmodium', 'protein'], 'euk_pro')

encode_and_dump(df.loc[df['Taxonomic lineage (GENUS)'] != 'Plasmodium', 'peptide'], 'euk_pep')
