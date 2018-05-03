#!/usr/bin/env bash


if [ $# != 1 ]; then
        echo "***ERROR*** Use: $0 <seq>"
        exit -1
fi


SEQ=$1
SEQNAME="${SEQ%.*}"
echo "Transforming $SEQ"
grep -v ">" $SEQ | tr '[:lower:]' '[:upper:]' | tr -d '\n' > $SEQNAME.fix.fasta
echo "Transformation completed"




