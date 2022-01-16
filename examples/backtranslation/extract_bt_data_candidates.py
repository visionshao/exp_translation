#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fileinput
from fairseq.models.transformer import TransformerModel
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract back-translations from the stdout of _fairseq-generate. "
            "If there are multiply hypotheses for a source sentence, we cat all of them with tab. "
        )
    )
    parser.add_argument("--output", required=True, help="output prefix")

    parser.add_argument(
        "--srclang", required=True, help="source language (extracted from H-* lines)"
    )
    parser.add_argument(
        "--tgtlang", required=True, help="target language (extracted from S-* lines)"
    )

    parser.add_argument("files", nargs="*", help="input files")
    args = parser.parse_args()

    def safe_index(toks, index, default):
        try:
            return toks[index]
        except IndexError:
            return default

    with open(args.output + "." + args.srclang, "w", encoding='utf-8') as src_h, \
            open(args.output + "." + args.srclang + ".bt_score", "w", encoding='utf-8') as src_score_h, \
            open(args.output + "." + args.srclang + ".bt_lprob", "w", encoding='utf-8') as src_lprob_h, \
            open(args.output + "." + args.tgtlang, "w", encoding='utf-8') as tgt_h:
        start = True
        for line in tqdm(fileinput.input(args.files, openhook=fileinput.hook_encoded('utf-8'))):
            if line.startswith("S-"):
                if not start:
                    print("\t".join(line_src), file=src_h)
                    print("\t".join(line_bt_score), file=src_score_h)
                    print("\t".join(line_bt_lprob), file=src_lprob_h)
                    print(tgt, file=tgt_h)
                start = False
                line_src = []
                line_bt_score = []
                line_bt_lprob = []
                tgt = safe_index(line.rstrip().split("\t"), 1, "")

            elif line.startswith("H-"):
                if tgt is not None:
                    splited_line = line.rstrip().split("\t")
                    score = safe_index(splited_line, 1, "")
                    src = safe_index(splited_line, 2, "")
                    line_src.append(src)
                    line_bt_score.append(score)
                    line_bt_lprob.append(str(eval(score) * (2 + len(src.split()))))


if __name__ == "__main__":
    main()
