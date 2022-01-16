from tqdm import tqdm
import fileinput
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="the input file prefix, should have .bt_lprob and .mono_lprob files")
    parser.add_argument("--output", required=True, help="the output file prefix, generate the .logim_score file")
    parser.add_argument("--silent_mode", default=True)
    args = parser.parse_args()
    with open(args.output + ".logim_score", 'w', encoding="utf-8") as output_h:
        for mono_line, bt_line in tqdm(
                zip(fileinput.input(args.input + ".mono_lprob", openhook=fileinput.hook_encoded("utf-8")),
                    fileinput.input(args.input + ".bt_lprob", openhook=fileinput.hook_encoded("utf-8"))),
                disable=args.silent_mode):
            mono_lprob = np.fromstring(mono_line, dtype=np.float, sep="\t")
            bt_lprob = np.fromstring(bt_line, dtype=np.float, sep="\t")
            log_im_score = mono_lprob - bt_lprob
            print("\t".join([str(x) for x in log_im_score.tolist()]), file=output_h)


if __name__ == "__main__":
    main()
