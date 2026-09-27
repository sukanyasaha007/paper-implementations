from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, get_weight_file_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('Helsinki-NLP/opus_books', f'{config["lang_src"]}-{config["lang_target"]}', split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_target'])

    split = ds_raw.train_test_split(test_size=0.1, seed=42)

    train_ds_raw = split["train"]
    val_ds_raw = split["test"]

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_target, config['lang_src'], config['lang_target'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_target, config['lang_src'], config['lang_target'], config['seq_len'])

    max_len_src = 0
    max_len_target = 0

    for item in ds_raw:
        # Each sentence becomes a list of token IDs, e.g. "I like cats" -> [45, 81, 209]
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_target.encode(item['translation'][config['lang_target']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_target}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target


def get_model(config, vocab_src_len, vocab_target_len):
    model = build_transformer(vocab_src_len, vocab_target_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_target, max_len, device):
    sos_idx = tokenizer_target.token_to_id('[SOS]')
    eos_idx = tokenizer_target.token_to_id('[EOS]')

    # precompute encoder output and reuse it for every token we get from the decoder - Prefill step
    encoder_output = model.encode(source, source_mask)
    #initialize decoder input with sos
    # use the decoder output and append the tokens generated for next decoder output - Decode Step
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim = 1)
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_target, max_len, device, print_msg, global_step, writer, num_examples= 2):
    model.eval()
    count = 0

    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_target, max_len, device)
            src_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_target.decode(model_out.detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f'Source: {src_text}')
            print_msg(f'Target: {target_text}')
            print_msg(f'Prediction: {model_out_text}')

            if count == num_examples:
                break


def train_model(config):
    # Select the GPU when available; model parameters and each batch must share this device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_target = get_ds(config)

    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_target.get_vocab_size(),
    ).to(device)

    # TensorBoard records loss values, while Adam updates all trainable model parameters.
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0

    # Resume model, optimizer, epoch, and logging state when a checkpoint is requested.
    if config['preload']:
        model_filename = get_weight_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']

    pad_token_id = tokenizer_src.token_to_id('[PAD]')
    if pad_token_id is None:
        raise ValueError('Source tokenizer does not contain [PAD]')
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=0.1,
    ).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(
            train_dataloader,
            desc=f'Processing epoch {epoch:02d}',
        )

        for batch in batch_iterator:
            model.train()
            # Move token IDs and masks from CPU memory to the model's device.
            encoder_input = batch['encoder_input'].to(device) #(batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) #(batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(batch, 1, 1, seq_len) - (batch, heads, query_seq_len, key_seq_len)
            decoder_mask = batch['decoder_mask'].to(device) #(batch, 1 seq_len, seq_len) - (batch, heads, query_seq_len, key_seq_len)
            '''
            encoder_mask.shape
            (batch, 1, 1, seq_len) - (8, 1, 1, 350)
            batch       = 8
            heads       = 1  → reuse mask for every attention head
            query       = 1  → reuse mask for every query token
            key position = 350

            Mask:             (8, 1, 1,   350)
            Attention scores: (8, 8, 350, 350)
                                    ↓
            Mask is reused across 8 heads and 350 query positions

            decoder_mask.shape
            # (batch, 1, seq_len, seq_len) - (8, 1, 350, 350)
            batch        = 8
            heads        = 1    → reuse mask for every attention head
            query tokens = 350
            key tokens   = 350
            '''

            # Encode the source, decode the shifted target, then predict target-vocabulary scores.
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                decoder_input,
                encoder_output,
                encoder_mask,
                decoder_mask,
            )
            projection_output = model.project(decoder_output)

            # Compare every predicted token distribution with its expected target token.
            label = batch['label'].to(device) # (B, seq_len)
            loss = loss_fn(
                projection_output.view(-1, tokenizer_target.get_vocab_size()),
                label.view(-1),
            )
            batch_iterator.set_postfix(loss=f'{loss.item():6.3f}')
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Clear old gradients, compute new gradients, and update the weights once.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # Print a few greedy translations at the end of every epoch.
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_target, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save enough state to resume training from the next epoch.
        model_filename = get_weight_file_path(config, f'{epoch:02d}')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
            },
            model_filename,
        )

    writer.close()


if __name__ == '__main__':
    train_model(get_config())
