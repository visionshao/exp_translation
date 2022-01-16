

import argparse
import fileinput
from fairseq.models.transformer import TransformerModel
from tqdm import tqdm
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract back-translations from the stdout of _fairseq-generate. "
            "If there are multiply hypotheses for a source sentence, we cat all of them with tab. "
        )
    )
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument("--lm_path", required=True, help="the mono language model checkpoint file path")
    parser.add_argument("--cp_file", help='the checkpoint file to load, none for best')
    parser.add_argument("--lm_data_path", required=True, help="the databin where the mono language model is trained")
    parser.add_argument("--bpe_file", required=False, help="the bpe code file")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="the batch size for mono language model scoring")
    parser.add_argument("--silent_mode", default=True)
    parser.add_argument("--max_positions", default=512,
                        help="the maximum number of positions, skip the sequences that are longer than this value.")

    parser.add_argument("files", nargs="*", help="input files")
    args = parser.parse_args()

    mono_model = TransformerModel.from_pretrained(
        args.lm_path,
        checkpoint_file=args.cp_file or 'checkpoint_best.pt',
        data_name_or_path=args.lm_data_path,
        bpe='fastbpe',
        bpe_codes=args.bpe_file or args.lm_data_path + '/code')

    mono_model.eval()  # disable dropout

    # Move model to GPU
    mono_model.cuda()

    with open(args.output + ".mono_lprob", "w", encoding='utf-8') as mono_lprob_h:
        n_batch = 0
        for line in tqdm(fileinput.input(args.files, openhook=fileinput.hook_encoded('utf-8')),
                         disable=args.silent_mode):
            line_src = line.split('\t')
            if n_batch == 0:
                n_batch = len(line_src) // args.batch_size
                if len(line_src) % args.batch_size != 0:
                    n_batch += 1
            line_mono_lprob = []
            for i in range(n_batch):
                batch_samples = line_src[i * args.batch_size:(i + 1) * args.batch_size]
                batch_samples = skip_invalid_samples(batch_samples, args.max_positions)
                batch_mono_lprob = []
                for x in mono_model.score(batch_samples):
                    lprob=x["positional_scores"].sum().cpu().numpy()
                    batch_mono_lprob.append(str(lprob))
                line_mono_lprob += batch_mono_lprob
            print("\t".join(line_mono_lprob), file=mono_lprob_h)


def skip_invalid_samples(batch_samples, max_position):
    output_batch = [x[:max_position] for x in batch_samples]
    return output_batch


if __name__ == "__main__":
    main()
