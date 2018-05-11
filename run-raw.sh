#!/bin/bash

QUERY=$1
REF=$2
DEV=$3
DIFF=$4
OVERLAP=$5

if [ $# != 5 ]; then
	echo "***ERROR*** Use: $0 query ref device diff overlap"
	exit -1
fi


indexA=$(basename "$QUERY")
indexB=$(basename "$REF")

BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PATHX=$(dirname "${QUERY}")
PATHY=$(dirname "${REF}")

SEQNAMEX="${indexA%.*}"
SEQNAMEY="${indexB%.*}"

$BINDIR/light_index_kmers -kwi 100 -query $QUERY -ref $REF -dev $DEV -diff $DIFF -olap $OVERLAP

Rscript --vanilla $BINDIR/plot_and_score.R ${indexA}-${indexB}.mat



