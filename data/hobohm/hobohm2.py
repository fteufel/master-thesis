#! /usr/bin/env python

import sys, os.path
from sets import Set

# Parse commandline, check that file name and cutoff is given and that file exists
# Note: should add support for stdin!!!
if not sys.argv[2:]:
	print("usage: %s CUT-OFF NEIGHBOR-LIST\n" % sys.argv[0])
	print("Hobohm2 algorithm for optimal homology-reduction\n")
	print("Input: a list of sequence similarities and a cutoff value.")
	print("       Sequences that are more similar than the cutoff, are")
	print("       considered to be neighbors.\n" )
	print("Output: a list of which sequences that should be kept in the data set\n")
	print("Format of NEIGHBOR-LIST:")
	print("       seqid_1 seqid_2 similarity")
	sys.exit()

cutoff = float(sys.argv[1])

if not os.path.isfile(sys.argv[2]):
	print "%s: file %s does not exist\n" % (sys.argv[0], sys.argv[2])
	sys.exit()

#######################################################################################
# Open neighbor-file, iterate over it one line at a time
# Read list of similarities, build list of names and keep track of neighbors.
# Neighbor info kept in dictionary, where key is seqname and value is Set of neighbors.
neighborfile = open(sys.argv[2], "r")
neighborlist = {}

# Each line in file has format: seqid_1 seqid_2 similarity
for simline in neighborfile:

	words=simline.split()
	if len(words)==3:                # Sanity check: do lines conform to expected format?
	    seq1=words[0]
	    seq2=words[1]
	    similarity=float(words[2])

	# Add sequence names as we go along
	if not neighborlist.has_key(seq1):
	    neighborlist[seq1]=Set()
	if not neighborlist.has_key(seq2):
	    neighborlist[seq2]=Set()

	# Build lists of neighbors as we go along.
	# Note: Set.add() method automatically enforces member uniqueness - saves expensive test!
	if similarity > cutoff:
	    neighborlist[seq1].add(seq2)
	    neighborlist[seq2].add(seq1)

neighborfile.close()

########################################################################################

# Build dictionary keeping track of how many neighbors each sequence has
nr_dict = {}
for seq in neighborlist.keys():
	nr_dict[seq]=len(neighborlist[seq])

# Find max number of neighbors
maxneighb = max(nr_dict.values())

# While some sequences in list still have neighbors: remove the one with most neighbors, update counts
# Note: could ties be dealt with intelligently?
while maxneighb > 0:

	# Find an entry that has maxneighb neighbors, and remove it from list
	for remove_seq in nr_dict.keys():
	    if nr_dict[remove_seq] == maxneighb: break
	del(nr_dict[remove_seq])

	# Update neighbor counts
	for neighbor in neighborlist[remove_seq]:
	    if neighbor in nr_dict:
        	nr_dict[neighbor] -= 1

	# Find new maximum number of neighbors
	maxneighb = max(nr_dict.values())

##############################################################################################
# Postprocess: reinstate skipped sequences that now have no neighbors
# Note: order may have effect. Could this be optimized?

allseqs=Set(neighborlist.keys())
keepseqs=Set(nr_dict.keys())
skipseqs=allseqs - keepseqs

for skipped in skipseqs:
	# if skipped sequence has no neighbors in keeplist
	if not (neighborlist[skipped] & keepseqs):
	    keepseqs.add(skipped)

# Print remaining sequences
for seq in keepseqs:
	print seq

