#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <K> <TRIES> <PDBS_TRAIN> <PDBS_VAL> <OUTPUT_DIR>"
    echo "  K           Number of discrete alphabet states"
    echo "  TRIES       Number of random seeds to evaluate"
    echo "  PDBS_TRAIN  File listing training PDB SIDs"
    echo "  PDBS_VAL    File listing validation PDB SIDs"
    echo "  OUTPUT_DIR  Directory to write final model and logs"
    exit 1
}

[ "$#" -eq 5 ] || usage

K=$1
TRIES=$2
PDBS_TRAIN=$3
PDBS_VAL=$4
OUTPUT_DIR=$5

RUN=${RUN:-}  # optional prefix, e.g. for job schedulers

THETA=270
TAU=0
D=2

mkdir -p tmp "$OUTPUT_DIR"

# Fetch PDBs
if [ ! -d tmp/pdb ]; then
    curl https://wwwuser.gwdg.de/~compbiol/foldseek/scp40pdb.tar.gz | tar -xz -C tmp
fi

# Build SSW aligner
if [ ! -f tmp/ssw_test ]; then
    git clone --depth 1 https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library tmp/ssw
    (cd tmp/ssw/src && make)
    cp tmp/ssw/src/ssw_test tmp/ssw_test
fi

# Filter TM-align pairfile to training PDBs only
awk 'FNR==NR {pdbs[$1]=1; next}
     ($1 in pdbs) && ($2 in pdbs) {print $1,$2,$10}' \
    "$PDBS_TRAIN" ../data/tmaln-06.out > tmp/pairfile_train.out

# Generate training features
uv run ../scripts/create_training_data.py \
    tmp/pdb tmp/pairfile_train.out $THETA $TAU $D tmp/vaevq_training_data.npy

# Train multiple seeds and benchmark each
for ((seed=0; seed<TRIES; seed++)); do
    echo -n "$seed " >> "$OUTPUT_DIR/log.txt"

    uv run ../scripts/train.py $seed tmp/vaevq_training_data.npy tmp "$K" \
        | awk '/opt_loss=/{printf "%s ", $2}' >> "$OUTPUT_DIR/log.txt"

    $RUN uv run ../scripts/encode_pdbs.py tmp/encoder.pt tmp/states.txt \
        --pdb_dir tmp/pdb --virt $THETA $TAU $D \
        < "$PDBS_TRAIN" > tmp/seqs.csv

    uv run ../scripts/create_submat.py tmp/pairfile_train.out tmp/seqs.csv \
        --mat tmp/sub_score.mat

    ./run-benchmark.sh tmp/encoder.pt tmp/states.txt tmp/sub_score.mat \
        "$PDBS_VAL" ../data/scop_lookup.tsv $THETA $TAU $D X >> "$OUTPUT_DIR/log.txt"
done

# Pick best seed (normalized against TMAalign reference AUCs)
SEED=$(awk '{print $1, ($3/0.928162 + $4/0.662063 + $5/0.275436) / 3}' "$OUTPUT_DIR/log.txt" \
    | sort -rk 2,2 | head -n 1 | awk '{print $1}')

# Train final model with best seed
uv run ../scripts/train.py "$SEED" tmp/vaevq_training_data.npy "$OUTPUT_DIR" "$K"

# Build final substitution matrix (includes all training PDBs)
awk 'FNR==NR {pdbs[$1]=1; next}
     ($1 in pdbs) && ($2 in pdbs) {print $1,$2,$10}' \
    "$PDBS_TRAIN" ../data/tmaln-06.out > tmp/pairfile_submat.out

$RUN uv run ../scripts/encode_pdbs.py "$OUTPUT_DIR/encoder.pt" "$OUTPUT_DIR/states.txt" \
    --pdb_dir tmp/pdb --virt $THETA $TAU $D \
    < "$PDBS_TRAIN" > tmp/seqs.csv

uv run ../scripts/create_submat.py tmp/pairfile_submat.out tmp/seqs.csv \
    --mat tmp/sub_score.mat --merge_state X \
    | tee tmp/create_submat.log

awk '/^assign_invalid_states_to/{printf "%s", $3}' \
    tmp/create_submat.log > "$OUTPUT_DIR/invalid_state.txt"

# Append X row/column to substitution matrix (scored as 0)
# TODO: adapt hardcoded X row length to K
awk 'NR==1 {printf "%s   X\n", $0}
     NR!=1  {printf "%s   0\n", $0}
     END    {print "X   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0"}' \
    tmp/sub_score.mat > "$OUTPUT_DIR/sub_score.mat"
