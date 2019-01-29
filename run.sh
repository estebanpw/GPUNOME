#!/bin/bash

QUERY=$1
REF=$2
DEV=$3
DIFF=$4
OVERLAP=$5
DIM=$6

if [ $# != 6 ]; then
	echo "***ERROR*** Use: $0 query ref device diff overlap dimension"
	exit -1
fi


indexA=$(basename "$QUERY")
indexB=$(basename "$REF")

BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PATHX=$(dirname "${QUERY}")
PATHY=$(dirname "${REF}")

SEQNAMEX="${indexA%.*}"
SEQNAMEY="${indexB%.*}"

if [ ! -f ${PATHX}/${SEQNAMEX}.fix.fasta ]; then

	$BINDIR/pre-process.sh $QUERY
fi

if [ ! -f ${PATHY}/${SEQNAMEY}.fix.fasta ]; then

	$BINDIR/pre-process.sh $REF
fi

$BINDIR/index_kmers_split_dyn_mat -kwi 100 -query ${PATHX}/${SEQNAMEX}.fix.fasta -ref ${PATHY}/${SEQNAMEY}.fix.fasta -dev $DEV -diff $DIFF -olap $OVERLAP -dim $DIM

Rscript --vanilla $BINDIR/plot_and_score.R ${SEQNAMEX}.fix.fasta-${SEQNAMEY}.fix.fasta.mat $DIM

#rm ${PATHX}/${SEQNAMEX}.fix.fasta
#rm ${PATHY}/${SEQNAMEY}.fix.fasta


