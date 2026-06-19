#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <ENCODER> <STATES> <SUBMAT> <PDBS> <SCOPLOOKUP> <THETA> <TAU> <D> <INVALID_STATE>"
    echo "  ENCODER       Path to encoder.pt"
    echo "  STATES        Path to states.txt"
    echo "  SUBMAT        Path to sub_score.mat"
    echo "  PDBS          File listing PDB SIDs to benchmark"
    echo "  SCOPLOOKUP    Path to scop_lookup.tsv"
    echo "  THETA TAU D   Virtual center parameters"
    echo "  INVALID_STATE Character used for invalid residues (e.g. X)"
    exit 1
}

[ "$#" -eq 9 ] || usage

ENCODER=$1
STATES=$2
SUBMAT=$3
PDBS=$4
SCOPLOOKUP=$5
THETA=$6
TAU=$7
D=$8
INVALID_STATE=$9

RUN=${RUN:-}

mkdir -p tmp/splits tmp/alignments

# Filter SCOP lookup to benchmark PDB set
awk 'FNR==NR {pdbs[$1]=1; next}
     ($1 in pdbs) {print $0}' \
    "$PDBS" "$SCOPLOOKUP" > tmp/scop_lookup_filtered.tsv

# Encode PDBs to 3Di sequences
$RUN uv run ../scripts/encode_pdbs.py "$ENCODER" "$STATES" \
    --pdb_dir tmp/pdb \
    --virt $THETA $TAU $D \
    --invalid-state "$INVALID_STATE" \
    < "$PDBS" > tmp/seqs.csv

cp "$SUBMAT" tmp/sub_score.mat

# Split query FASTA for parallel Smith-Waterman
awk '{print ">" $1} {print $2}' < tmp/seqs.csv > tmp/target.fasta
split -n 30 -d tmp/target.fasta tmp/splits/split_ --additional-suffix=.fasta

./run-smithwaterman.sh 8 2

# Score alignments against SCOP hierarchy
./roc1.awk tmp/scop_lookup_filtered.tsv \
    <(cat tmp/alignments/*.m8) > tmp/result.rocx

# Print mean AUC at family / superfamily / fold level
awk '{famsum+=$3; supfamsum+=$4; foldsum+=$5}
     END {print famsum/NR, supfamsum/NR, foldsum/NR}' \
    tmp/result.rocx
