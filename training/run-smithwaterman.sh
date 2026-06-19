#!/bin/bash
set -eo pipefail

usage() {
    echo "Usage: $0 <GAP_OPEN> <GAP_EXTEND>"
    echo "  Runs parallel Smith-Waterman over tmp/splits/ against tmp/target.fasta"
    echo "  Reads:  tmp/sub_score.mat, tmp/splits/*.fasta"
    echo "  Writes: tmp/alignments/*.m8"
    exit 1
}

[ "$#" -eq 2 ] || usage

GAP_OPEN=$1
GAP_EXTEND=$2

# Semaphore: limits concurrent SSW jobs to N
open_sem() {
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for ((; i > 0; i--)); do printf %s 000 >&3; done
}

run_with_lock() {
    local x
    read -u 3 -n 3 x && ((0 == x)) || exit $x
    (
        ("$@")
        printf '%.3d' $? >&3
    ) &
}

task() {
    tmp/ssw_test -o "$GAP_OPEN" -e "$GAP_EXTEND" -a tmp/s.mat -p \
        -f 50 tmp/target.fasta "$1" \
        2>/dev/null \
    | mawk '/^target/{target=$2}
            /^query/{query=$2}
            /^optimal_alignment_score/{score=$2; print query,target,score}' \
    | sort -k1,1 -k3,3nr \
    > "tmp/alignments/${1##*/}.m8"
}

mkdir -p tmp/alignments

# Matrix filename must be ≤16 chars for SSW
cp tmp/sub_score.mat tmp/s.mat

N=64
open_sem $N
for split in tmp/splits/*; do
    run_with_lock task "$split"
done

# Wait for all SSW jobs to finish
wait
