#!/bin/bash

if [ $# -ne 5 ]; then
    echo "usage: $0 [dataset=wmt14/full] [langpair=en-de] [databin] [bpecode] [model]"
    exit
fi


DATASET=$1
LANGPAIR=$2
DATABIN=$3
BPECODE=$4
MODEL=$5

SRCLANG=$(echo $LANGPAIR | cut -d '-' -f 1)
TGTLANG=$(echo $LANGPAIR | cut -d '-' -f 2)


BPEROOT=/apdcephfs/share_916081/visionshao/subword-nmt/subword_nmt

if [ ! -e $BPEROOT ]; then
    BPEROOT=subword-nmt/subword_nmt
    if [ ! -e $BPEROOT ]; then
        echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
        git clone https://github.com/rsennrich/subword-nmt.git
    fi
fi

sacrebleu -t $DATASET -l $LANGPAIR --echo src \
| sacremoses tokenize -a -l $SRCLANG -q \
| python3 $BPEROOT/apply_bpe.py -c $BPECODE \
| fairseq-interactive $DATABIN --path $MODEL \
    -s $SRCLANG -t $TGTLANG \
    --beam 5 --remove-bpe --buffer-size 1024 --max-tokens 1024 \
| grep ^H- | cut -f 3- \
| sacremoses detokenize -l $TGTLANG -q \
| sacrebleu -t $DATASET -l $LANGPAIR
