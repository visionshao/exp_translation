from tqdm import tqdm
import fileinput
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_files", nargs="*", required=True, help="the candidate files")

    parser.add_argument("--score_files", nargs="*", required=True,
                        help="the score file corresponding to the candidate file")

    parser.add_argument("--strategy", required=True,
                        help="the strategy to extract the back-translation data, should be highest or lowest to "
                             "select the candidate based on highest or lowest importance score")

    parser.add_argument("--output", required=True, help="the output extracted file prefix")

    args = parser.parse_args()
    assert args.strategy in ['highest', 'lowest'], \
        f"argument error:{args.strategy} is not highest or lowest"


    with open(args.output + ".de", 'w', encoding="utf-8") as output_h:
        if args.strategy == 'highest':
            for candidate_line, score_line in tqdm(
                    zip(fileinput.input(args.candidate_files, openhook=fileinput.hook_encoded("utf-8")),
                        fileinput.input(args.score_files, openhook=fileinput.hook_encoded("utf-8")))):
                weights = np.fromstring(score_line, sep='\t')
                try:
                    selected_candidate = candidate_line.rstrip().split('\t')[np.nanargmax(weights)]
                except:
                    selected_candidate = ""
                print(selected_candidate, file=output_h)

        elif args.strategy == 'lowest':
            for candidate_line, score_line in tqdm(
                    zip(fileinput.input(args.candidate_files, openhook=fileinput.hook_encoded("utf-8")),
                        fileinput.input(args.score_files, openhook=fileinput.hook_encoded("utf-8")))):
                weights = np.fromstring(score_line, sep='\t')
                try:
                    selected_candidate = candidate_line.rstrip().split('\t')[np.nanargmin(weights)]
                except:
                    selected_candidate = ""
                print(selected_candidate, file=output_h)
        else:
            raise NotImplementedError(f"argument error:{args.strategy} is not highest or lowest")


if __name__ == "__main__":
    main()
