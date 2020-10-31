'''
Utilities to process deterministic SignalP labels
[S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]
to arrays of non-deterministic labels with region information


The SP starts with the n-region that is composed of a positively charged stretch of 5–8 amino acid residues.
This part probably enforces the proper topology on a polypeptide during its translocation through the endoplasmic 
reticulum membrane based on the positive-inside rule [9]. The n-region is followed by the h-region, a stretch of 
8–12 hydrophobic amino acids, which constitutes the core of the SP and usually forms an α-helix. The third component 
of the SP is the polar and uncharged c-region. This part is usually six residues long and ends with a cleavage site,
at which a signal peptidase cuts the SP off during or after protein translocation into the lumen of the endoplasmic reticulum [10].

http://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/Different_types_signal_peptides.html

https://mbio.asm.org/content/8/4/e00909-17




'''

import numpy as np
from typing import Dict, Tuple
from math import ceil

#import sys
#sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #define proper __init__.py files sometime later
#import ipdb; ipdb.set_trace()
#from resources.labeling_resources import KYTE_DOOLITTLE, SP_REGION_VOCAB

KYTE_DOOLITTE = {'I':  4.5, 
                 'V':  4.2,
                 'L':  3.8,
                 'F':  2.8,
                 'C':  2.5,
                 'M':  1.9,
                 'A':  1.8,
                 'G': -0.4,
                 'T': -0.7,
                 'S': -0.8,
                 'W': -0.9,
                 'Y': -1.3,
                 'P': -1.6,
                 'H': -3.2,
                 'E': -3.5,
                 'Q': -3.5,
                 'D': -3.5,
                 'N': -3.5,
                 'K': -3.9,
                 'R': -4.5}


#multi-state label dict

SP_REGION_VOCAB = {
                    'NO_SP_I' : 0,
                    'NO_SP_M' : 1,
                    'NO_SP_O' : 2,

                    'SP_N' :    3,
                    'SP_H' :    4,
                    'SP_C' :    5,
                    'SP_I' :    6,
                    'SP_M' :    7,
                    'SP_O' :    8,

                    'LIPO_N':   9,
                    'LIPO_H':  10,
                    'LIPO_CS': 11, #conserved 2 positions before the CS are not hydrophobic,but are also not considered a c region
                    'LIPO_C1': 12, #the C in +1 of the CS
                    'LIPO_I':  13,
                    'LIPO_M':  14,
                    'LIPO_O':  15,

                    'TAT_N' :  16,
                    'TAT_RR':  17, #conserved RR marks the border between n,h
                    'TAT_H' :  18,
                    'TAT_C' :  19,
                    'TAT_I' :  20,
                    'TAT_M' :  21,
                    'TAT_O' :  22,

                    'PILIN_P': 23,
                    'PILIN_CS':24,
                    'PILIN_H': 25,
                    }


def apply_sliding_hydrophobicity_window(sequence: np.ndarray, window_size =7) -> int:
    '''Run a window of window_size over sequence and compute hydrophobicity.
    Uses the kyte-doolittle scale.
    Return index of center residue within the most hydrophobic window'''
    best_score = -100000000000000000
    most_hydrophobic_pos = None
    for start_idx in range(len(sequence)-window_size):
        subseq = sequence[start_idx:start_idx+7]
        hydropathy_score = sum([KYTE_DOOLITTE[res] for res in subseq])
        #TODO define tie-breaking behavior: > or => ?
        if hydropathy_score>best_score:
            best_score = hydropathy_score
            most_hydrophobic_pos = start_idx + ceil(window_size/2) #position of mid residue in window

    return most_hydrophobic_pos

def find_twin_arginine(sequence: str) -> Tuple[int,int]:
    '''Find the last RR in sequence. If no RR, tries KR.
    If no KR, takes last R and considers +1 and -1 of it RR states'''
    #TODO h,n need to know when labels get probabilistic
    last_rr_start =  sequence.rfind('RR')
    last_rr_end = last_rr_start +1

    if last_rr_start == -1:
        #resort to less stringent motif
        last_rr_start = sequence.rfind('KR')
        last_rr_end = last_rr_start +1
    if last_rr_start == -1:
        #can't decide for sure, use noisy labels
        last_r = sequence.rfind('R')
        last_rr_start = last_r -1
        last_rr_end = last_r+1

    return last_rr_start, last_rr_end

    
def process_SP(label_sequence: str, aa_sequence: str, sp_type=str, vocab: Dict[str,int]= None) -> np.ndarray:
    '''Generate multi-state tag array from SignalP label string and sequence
    '''
    if vocab is None:
        vocab = SP_REGION_VOCAB

    #make matrix to fill up
    tag_matrix = np.zeros((len(label_sequence), len(vocab)))
    #find end of SP and convert sequence str to list of AAs
    type_tokens = {'NO_SP': 'I', 'SP': 'S', 'LIPO': 'L', 'TAT':'T'}
    last_idx = label_sequence.rfind(type_tokens[sp_type]) #TODO needs to be based on sp_type

    sp = [x for x in aa_sequence[:last_idx+1]]
    sp = np.array(sp)

    label_array = np.array([x for x in label_sequence])

    transmembrane_idx = np.where(label_array == 'M')
    extracellular_idx = np.where(label_array == 'O')
    intracellular_idx = np.where(label_array == 'I')

    #find most hydrophobic position
    hydrophobic_pos = apply_sliding_hydrophobicity_window(sp)

    if sp_type=='NO_SP':
        tag_matrix[intracellular_idx, vocab['NO_SP_I']] = 1
        tag_matrix[transmembrane_idx, vocab['NO_SP_M']] = 1
        tag_matrix[extracellular_idx, vocab['NO_SP_O']] = 1

    elif sp_type == 'SP':
        n_end = hydrophobic_pos
        h_start = 2
        h_end = last_idx-2
        c_start = hydrophobic_pos+1

        tag_matrix[:n_end, vocab['SP_N']] =1
        tag_matrix[h_start:h_end, vocab['SP_H']] =1
        tag_matrix[c_start:last_idx+1, vocab['SP_C']] =1

    elif sp_type == 'LIPO':
        tag_matrix[intracellular_idx, vocab['LIPO_I']] = 1
        tag_matrix[transmembrane_idx, vocab['LIPO_M']] = 1
        tag_matrix[extracellular_idx, vocab['LIPO_O']] = 1

        n_end = hydrophobic_pos
        h_start = 2
        h_end = last_idx-1


        tag_matrix[:n_end, vocab['LIPO_N']] =1
        tag_matrix[h_start:h_end, vocab['LIPO_H']] =1

        tag_matrix[last_idx-1:last_idx+1, vocab['LIPO_CS'] ] = 1
        tag_matrix[last_idx+1, :] = 0 #override O,I,M that was set here before
        tag_matrix[last_idx+1, vocab['LIPO_C1']] = 1



    elif sp_type == 'TAT':
        #find last two arginines in sp section
        last_rr_start, last_rr_end = find_twin_arginine(aa_sequence[:last_idx+1])
        #last_rr_start =  aa_sequence[:last_idx+1].rfind('RR')
        len_motif = last_rr_end - last_rr_start # check whether I have the real motif(2) or unsure(3)
        if len_motif ==2:
            n_end = last_rr_start
            h_start = last_rr_end+2
        if len_motif ==3:
            n_end = last_rr_start+1
            h_start = last_rr_end

        h_end = last_idx-1
        c_start = hydrophobic_pos+1

        tag_matrix[:last_rr_start, vocab['TAT_N']] =1
        tag_matrix[last_rr_start:last_rr_start+2, vocab['TAT_RR']] =1
        tag_matrix[last_rr_start+2, vocab['TAT_N']] = 1 #pos after rr motif is n
        tag_matrix[last_rr_start+3:h_end, vocab['TAT_H']] =1
        tag_matrix[c_start:last_idx+1, vocab['TAT_C']] = 1

        tag_matrix[intracellular_idx, vocab['TAT_I']] = 1
        tag_matrix[transmembrane_idx, vocab['TAT_M']] = 1
        tag_matrix[extracellular_idx, vocab['TAT_O']] = 1
        
    elif sp_type =='PILIN':
        #hydrophobic region is after CS
        hydrophobic_pos = apply_sliding_hydrophobicity_window(np.array([x for x in aa_sequence[last_idx:last_idx+20]]))

        motif_end = last_idx+6 #first 5 after SP are conserved motif
        tag_matrix[:last_idx+1, vocab['PILIN_P']] =1
        tag_matrix[last_idx+1:motif_end, vocab['PILIN_CS']] = 1
        tag_matrix[motif_end, vocab['PILIN_H']] = 1 #
        #TODO end of H?

        tag_matrix[intracellular_idx, vocab['PILIN_I']] = 1
        tag_matrix[transmembrane_idx, vocab['PILIN_M']] = 1
        tag_matrix[extracellular_idx, vocab['PILIN_O']] = 1

    else:
        raise NotImplementedError(f'Unknown type {sp_type}')


    #quality control: at least 1 state active at each position
    assert not any(tag_matrix.sum(axis=1) ==0), 'Processing failed. There are positions where no state was set active.'
    return tag_matrix
