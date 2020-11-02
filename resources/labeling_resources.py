'''
static resources used for generating training labels
'''
#hydropathy scale
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

