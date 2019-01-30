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

filename=$(basename -- "$QUERY")
extensionA="${filename##*.}"

filename=$(basename -- "$REF")
extensionB="${filename##*.}"

indexA=$(basename "$QUERY")
indexB=$(basename "$REF")

BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PATHX=$(dirname "${QUERY}")
PATHY=$(dirname "${REF}")

SEQNAMEX="${indexA%.*}"
SEQNAMEY="${indexB%.*}"

if [ ! -f ${PATHX}/${SEQNAMEX}.${extensionA}.fix ]; then

	$BINDIR/pre-process.sh $QUERY
fi

if [ ! -f ${PATHY}/${SEQNAMEY}.${extensionB}.fix ]; then

	$BINDIR/pre-process.sh $REF
fi

$BINDIR/index_kmers_split_dyn_mat -kwi 100 -query ${PATHX}/${SEQNAMEX}.${extensionA}.fix -ref ${PATHY}/${SEQNAMEY}.${extensionB}.fix -dev $DEV -diff $DIFF -olap $OVERLAP -dim $DIM

# Remove the fix portion
mv ${SEQNAMEX}.${extensionA}.fix-${SEQNAMEY}.${extensionB}.fix.mat ${SEQNAMEX}.${extensionA}-${SEQNAMEY}.${extensionB}.mat

(Rscript --vanilla $BINDIR/plot_and_score.R ${SEQNAMEX}.${extensionA}-${SEQNAMEY}.${extensionB}.mat $DIM) &> ${SEQNAMEX}.${extensionA}-${SEQNAMEY}.${extensionB}.scr.txt

# (Rscript $BINDIR/compute_score.R $FILE1-$FILE2.mat $DIM) &> $FILE1-$FILE2.scr.txt

#rm ${PATHX}/${SEQNAMEX}.fix.fasta
#rm ${PATHY}/${SEQNAMEY}.fix.fasta


