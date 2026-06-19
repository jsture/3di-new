#!/bin/bash
set -euo pipefail

# 4-fold cross-validation: train on 3 folds, benchmark on held-out fold.
# Compare split models against the reference foldseek_v1 alphabet.
#
# Usage: RUN=<prefix> ./crossval.sh
# Reads:  data/scop_lookup.tsv, data/foldseek_v1/
# Writes: tmp/crossval.log, tmp/crossval_{ref,splitmodel}<k>.rocx

K=20     # alphabet size for split models
TRIES=10 # seeds per fold

RUN=${RUN:-}

mkdir -p tmp

# Split SCOP domain list into 4 partitions by fold
uv run ../scripts/split_folds.py --lookup-file data/scop_lookup.tsv --out-dir tmp

# Build 4x train/val PDB lists from fold splits
rm -f tmp/pdbs_train?.txt tmp/pdbs_val?.txt

for k in {0..3}; do
    for s in {0..3}; do
        if [ "$k" = "$s" ]; then
            awk '/^d/{print $1}' "tmp/fold_split$s.txt" >> "tmp/pdbs_val$k.txt"
        else
            awk '/^d/{print $1}' "tmp/fold_split$s.txt" >> "tmp/pdbs_train$k.txt"
        fi
    done
done

# Train a split model for each fold
for k in {0..3}; do
    RUN=$RUN ./learnAlphabet.sh "$K" "$TRIES" \
        "tmp/pdbs_train$k.txt" "tmp/pdbs_val$k.txt" "tmp/splitmodels/sp$k"
done

# Benchmark each split model against the reference foldseek_v1 alphabet
for k in {0..3}; do
    echo "Fold $k" >> tmp/crossval.log

    echo -n "Ref: " >> tmp/crossval.log
    RUN=$RUN ./run-benchmark.sh \
        data/foldseek_v1/encoder.pt \
        data/foldseek_v1/states.txt \
        data/foldseek_v1/sub_score.mat \
        "tmp/pdbs_val$k.txt" data/scop_lookup.tsv 270 0 2 D \
        | tee -a tmp/crossval.log
    cp tmp/result.rocx "tmp/crossval_ref$k.rocx"

    INVST=$(cat "tmp/splitmodels/sp$k/invalid_state.txt")
    echo -n "Splitmodel: " >> tmp/crossval.log
    RUN=$RUN ./run-benchmark.sh \
        "tmp/splitmodels/sp$k/encoder.pt" \
        "tmp/splitmodels/sp$k/states.txt" \
        "tmp/splitmodels/sp$k/sub_score.mat" \
        "tmp/pdbs_val$k.txt" data/scop_lookup.tsv 270 0 2 "$INVST" \
        | tee -a tmp/crossval.log
    cp tmp/result.rocx "tmp/crossval_splitmodel$k.rocx"
done
