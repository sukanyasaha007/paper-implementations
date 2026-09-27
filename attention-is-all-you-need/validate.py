"""Load a trained checkpoint and print greedy translations from the validation set.

Usage: python validate.py [epoch]   (default: latest epoch from config, e.g. 19)
"""
import sys

import torch

from config import get_config, get_weight_file_path
from train import get_ds, get_model, run_validation


def main(epoch: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config()
    _, val_dataloader, tokenizer_src, tokenizer_target = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    model_filename = get_weight_file_path(config, epoch)
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    run_validation(model, val_dataloader, tokenizer_src, tokenizer_target, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples = 10)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else f"{get_config()['num_epochs'] - 1:02d}")
