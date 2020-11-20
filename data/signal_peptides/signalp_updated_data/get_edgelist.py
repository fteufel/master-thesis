import subprocess
from os import path

edgelist_out = 'edgelist_rerun/all_aligned.csv'
#when self-aligning huge datasets, use multiple batches. query is subset of data, lib is full data.
#cat the files after processing, graph_part does not care about order when parsing.
fasta_file_query = '/work3/felteu/edgelist_splitted/signalp6_seqonly_for_graphpart.fasta'#full_updated_data_seqs_only.fasta'
fasta_file_lib = '/work3/felteu/edgelist_splitted/signalp6_seqonly_for_graphpart.fasta'
ggs = path.expanduser('/work3/felteu/fasta36/bin/ggsearch36')

with subprocess.Popen(
        [ggs,"-E","41762",fasta_file_query,fasta_file_lib],
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True) as proc:
    with open(edgelist_out,'w+') as outf:
        for line_nr, line in enumerate(proc.stdout):
            if '>>>' in line:
                qry_nr = int(line[2])
                this_qry = line[6:70].split()[0].split('|')[0]

            elif line[0:2] == '>>':
                this_lib = line[2:66].split()[0].split('|')[0]

            elif line[:13] == 'global/global':
                identity = float(line.split()[4][:-1])/100
                #print(qry_nr, this_qry, this_lib, identity)

                if this_qry == this_lib:
                    continue
                outf.write("%s,%s,%.3f\n" % (this_qry, this_lib,identity))