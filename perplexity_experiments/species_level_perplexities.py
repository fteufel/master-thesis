'''
Evaluate perplexities separately per species
'''
import sys
sys.path.append('..')
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from models.awd_lstm import ProteinAWDLSTMForLM, ProteinAWDLSTMConfig
from tape import TAPETokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tape.datasets import pad_sequences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Evaluation of LMs on per-species level.')
parser.add_argument('--data', type=str, default='data/awdlstmtestdata/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint_dir', type=str, default='/zhome/1d/8/153438/experiments/results/',
                    help='location of the saved model')
parser.add_argument('--output_dir', type = str, default='perplexity_test',
                    help='output directory')
args = parser.parse_args()

test_data = pd.read_csv(args.data, sep = '\t')
test_data.head()

organisms = test_data['Organism'].value_counts()
organisms = organisms[organisms>10]

class FullSeqSeriesDataSet(torch.utils.data.Dataset):
    def __init__(self, pd_series):
        super().__init__
        self.data = pd_series.values
        self.tokenizer = TAPETokenizer()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        tokenized = self.tokenizer.tokenize(seq) + [self.tokenizer.stop_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        input_ids = token_ids[:-1]
        target_ids =token_ids[1:]
        assert len(target_ids) == len(input_ids)
        return np.array(input_ids), np.array(target_ids) 
    
    @staticmethod
    def collate_fn(batch):
        data, targets = tuple(zip(*batch))   
        torch_data = torch.from_numpy(pad_sequences(data, 0)) #0 is tokenizer pad token
        torch_targets = torch.from_numpy(pad_sequences(targets, -1)) #pad with -1 to ignore loss

        return torch_data.permute(1,0), torch_targets.permute(1,0)  # type: ignore
    
def run_model_on_data(model, dataset) -> float:
    dl = torch.utils.data.DataLoader(ds, batch_size=100, collate_fn=ds.collate_fn)
    total_loss = 0
    for i, batch in enumerate(dl):
        data, targets = batch
        data = data.to(device).contiguous()
        targets = targets.to(device).contiguous()
        with torch.no_grad():
            loss, _, _ = model(data, targets = targets) #loss, output, hidden states        
        #NOTE this is the mean loss over all dimensions: batch_size, seq_len
        #Here I am using datasets with a low number of sequences, so the last batch will be much less than 100 presumably.
        #To compare, need to make sure I do not ignore this -> no sum(avg_loss)/n_batches mean.
        #For the model comparison this was no issue, as all models were run on the same dataset, so the error would be systematic
        #If it is even an error. No best practice on how to really average perplexity.
        total_loss += loss.item()*len(data)
        
    return total_loss / len(dataset)


model = ProteinAWDLSTMForLM.from_pretrained(args.checkpoint_dir)

results_dict = {}
len_dict = {}
for org in tqdm(organisms.index):
    print(org)
    data =  test_data.loc[test_data['Organism'] ==org, 'Sequence']
    ds= FullSeqSeriesDataSet(data)
    len_dict[org] = len(ds)
    loss = run_model_on_data(model, ds)
    results_dict[org] = np.exp(loss)
    
    
#make df

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

df_out = pd.DataFrame([pd.Series(results_dict), pd.Series(lendict)], index = ['perplexity', 'n_sequences']).T

df_out.to_csv(os.path.join(args.output_dir, 'perplexities_perspecies.csv'))
