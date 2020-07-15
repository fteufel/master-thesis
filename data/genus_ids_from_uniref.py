
import argparse
import gzip
import urllib.parse
import urllib.request
import pandas as pd
import os
from multiprocessing import Process, Pipe


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
    
    n = 20

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
            from IPython import embed
            embed()
            for line in f:
                line = line.decode('utf-8').strip().split('\t')
                #need to assert validity for each entry,deal with missing ifnrmation or entry not in uniprot
                if len(line)<3:
                    print(f'broken record {line[0]}')
                    genus, id, kingdom = line[0], 'NA', 'NA'
                else:
                    genus, id, kingdom = line[0], line[1], line[2]#' '.join(list(line[0]))
                formated_file.write('%s\t%s\t%s\n' % (genus, id, kingdom))

        i += 1
    formated_file.close()


def fix_download_problems(id_list, decoded_split_lines):
    '''Correct for IDs that could not be downloaded. Uniprot API just ignores them, no errors.'''
    num_ids = len(id_list)
    num_lines = len(decoded_split_lines)
    fixed_lines = []
    i = 0
    current_id_idx = 0
    current_line_idx = 0
    while i<num_ids:
        current_line = decoded_split_lines[current_line_idx] if current_line_idx<num_lines else ["NA","NA","NA"] #dumb solution for end of iteration
        if current_line[-1] == id_list[current_id_idx]:
            #line matches. append and move ahead in both lists.
            fixed_lines.append(current_line)
            current_id_idx +=1
            current_line_idx +=1 
        else:
            #line does not match. Move id.
            fixed_lines.append(["NA", "NA", id_list[current_id_idx]])
            current_id_idx +=1
        i +=1
    return fixed_lines


def download_ids(ids):
    url = 'https://www.uniprot.org/uploadlists/'
    query = ids

    tmp_query = ' '.join(query)
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

    with urllib.request.urlopen(req) as f:
        f.readline() #skips the header
        lines = []
        for line in f:
            line = line.decode('utf-8').strip().split('\t')
            lines.append(line)

        lines = fix_download_problems(ids, lines)
    
    return lines
        


### Remote download worker
def worker(remote, parent_remote):
    parent_remote.close()
    while True:
        cmd, data = remote.recv()
        if cmd == 'download':
            success = False
            while success == False:
                try:
                    lines = download_ids(data)
                    success = True
                except:
                    print('A Connection failed, retrying')
            remote.send(lines)
        else:
            raise NotImplementedError

def download_set_multiprocess(ids):
    query = ids
    #make remote workers
    waiting = False
    closed = False

    n_workers = 20

    remotes, work_remotes = zip(*[Pipe() for _ in range(n_workers)])
    processes = [Process(target=worker, args=(work_remote, remote))
                for (work_remote, remote) in zip(work_remotes, remotes)]
    for p in processes:
        p.daemon = True # if the main process crashes, we should not cause things to hang
        p.start()
    for remote in work_remotes:
        remote.close()


    n_total = len(query)
        
    n = 3000

    formated_file = open(args.output_file, 'w')


    i = 0
    while i*n*n_workers < n_total:


        tmp_query = query[ i*n*n_workers : i*n*n_workers+n*n_workers ] #get e.g. 30k indices
        #make indices
        dl_ids = [ tmp_query[i*n:i*n+n] for i in range(n_workers)]


        for remote, ids in zip(remotes, dl_ids):
            remote.send(('download', ids))
        waiting = True


        results = [remote.recv() for remote in remotes] #list of lists
        waiting = False

        #flatten lines
        all_lines = [item for sublist in results for item in sublist]



        for line in all_lines:
            if len(line)<3:
                print(f'broken record {line}')
                genus, kingdom, id = 'NA', 'NA', line[-1]
            else:
                genus, kingdom, id = line[0], line[1], line[2]#' '.join(list(line[0]))

            formated_file.write('%s\t%s\t%s\n' % (genus, kingdom, id))


        i+=1
        print(f'Iteration {i}, got {i*n*n_workers} records')





download_set_multiprocess(id_list)
