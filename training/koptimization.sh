#!/bin/bash
set -euo pipefail

# Grid search over alphabet size K=4..40 (step 4).
# Patch SSW first to support alphabets larger than 20 states.
#
# Usage: RUN=<prefix> ./koptimization.sh
# Reads:  data/pdbs_train.txt, data/pdbs_val.txt, data/scop_lookup.tsv
# Writes: tmp/kmodels/<k>/, tmp/koptimization.log

TRIES=20
RUN=${RUN:-}

mkdir -p tmp

# Patch and rebuild SSW to handle larger alphabet sizes
patch -u tmp/ssw/src/ssw.c -i ssw.patch
(cd tmp/ssw/src && make && cp ssw_test ../../ssw_test)

# Train one model per K value
for k in {4..40..4}; do
    RUN=$RUN ./learnAlphabet.sh "$k" "$TRIES" \
        data/pdbs_train.txt data/pdbs_val.txt "tmp/kmodels/$k"
done

# Combine train+val for final benchmark
sort -u data/pdbs_train.txt data/pdbs_val.txt > tmp/pdbs_all.txt

# Benchmark each K model
for k in {4..40..4}; do
    echo -n "$k " >> tmp/koptimization.log
    INVST=$(cat "tmp/kmodels/$k/invalid_state.txt")
    RUN=$RUN ./run-benchmark.sh \
        "tmp/kmodels/$k/encoder.pt" \
        "tmp/kmodels/$k/states.txt" \
        "tmp/kmodels/$k/sub_score.mat" \
        tmp/pdbs_all.txt data/scop_lookup.tsv 270 0 2 "$INVST" \
        | tee -a tmp/koptimization.log
    cp tmp/result.rocx "tmp/kmodels/$k/result.rocx"
done
