#!/usr/bin/env bash


if [ $# != 1 ]; then
        echo "***ERROR*** Use: $0 <seq>"
        exit -1
fi


SEQ=$1
SEQNAME="${SEQ%.*}"
filename=$(basename -- "$SEQ")
extension="${filename##*.}"

echo "Transforming $SEQ"
grep -v ">" $SEQ | tr '[:lower:]' '[:upper:]' | tr -d '\n' | sed "s/[^ACGT]/N/g"  > $SEQNAME.${extension}.fix
echo "Transformation completed"




