
import argparse
import gzip
import urllib.parse
import urllib.request
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/', help='UniRef tab separated dump file')
parser.add_argument('--output_file', type=str, default='genus_ids.tsv')

args = parser.parse_args()
df = pd.read_csv(args.data, usecols= ['Cluster ID'], sep = '\t')
prefix = 'UniRef90_' #remove the uniref prefix to get the id of the representative sequence
id_list = list(df['Cluster ID'].str.lstrip(prefix))

def download_set(ids):
    url = 'https://www.uniprot.org/uploadlists/'
    query = ids #[line.rstrip('\n').lstrip(prefix) for line in gzip.open('%s/%s_%s_%s/%s_ids.txt.gz' % (loc,domain,complete,quality,dataset), 'rt')]
    
    n_total = len(query)
    
    n = 3000

    formated_file = open(args.output_file, 'w')
    
    i = 0
    while i*n < n_total:
        tmp_query = query[i*n:i*n+n]
        tmp_query = ' '.join(tmp_query)
        params = {
        'from': 'ACC+ID',
        'to': 'ACC',
        'format': 'tab',
         'columns': 'lineage(GENUS),lineage(SUPERKINGDOM)',
        'query': tmp_query
        }
        
        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)

        print(f'opening url {i}')
        with urllib.request.urlopen(req) as f:
            f.readline() #skips the header
            for line in f:
                line = line.decode('utf-8').strip().split('\t')
                if len(line)<3:
                    print(f'broken record {line[0]}')
                    genus, id, kingdom = line[0], None, None
                else:
                    genus, id, kingdom = line[0], line[1], line[2]#' '.join(list(line[0]))
                formated_file.write('%s\t%s\t%s\n' % (genus, id, kingdom))

        i += 1
    formated_file.close()

download_set(id_list)