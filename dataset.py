import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader


class MedicalDataset(Dataset):

    def __init__(self, sequences, labels, vocab, max_len=512):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        seq = self.sequences[idx]
        label = self.labels[idx]

        seq_ids = [self.vocab.get(token, self.vocab["[UNK]"]) for token in seq]

        if len(seq_ids) > self.max_len - 1:
            seq_ids = seq_ids[-(self.max_len - 1):]

        seq_ids = [self.vocab["[CLS]"]] + seq_ids

        pad_len = self.max_len - len(seq_ids)

        mask = [1] * len(seq_ids) + [0] * pad_len
        seq_ids += [self.vocab["[PAD]"]] * pad_len

        return (
            torch.tensor(seq_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


def train_test_split_custom(seqs, labels, test_size=0.2, random_state=42):

    random.seed(random_state)

    combined = list(zip(seqs, labels))
    random.shuffle(combined)

    split_idx = int(len(combined) * (1 - test_size))

    train = combined[:split_idx]
    test = combined[split_idx:]

    seq_train, y_train = zip(*train)
    seq_test, y_test = zip(*test)

    return list(seq_train), list(seq_test), list(y_train), list(y_test)


def build_vocab(train_sequences, min_freq=1):

    counter = Counter()

    for seq in train_sequences:
        counter.update(seq)

    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[REPORT_START]": 4
    }

    idx = len(vocab)

    for token, freq in counter.items():
        if freq < min_freq: continue
        if token in vocab: continue

        vocab[token] = idx
        idx += 1

    return vocab


def get_dataloaders(json_path, batch_size=32, max_len=512):

    (
        subject_ids,
        hadm_ids,
        sequences,
        labels
    ) = load_dataset_json(json_path)

    seq_train, seq_temp, y_train, y_temp = train_test_split_custom(
        sequences,
        labels,
        test_size=0.2,
        random_state=42
    )

    seq_val, seq_test, y_val, y_test = train_test_split_custom(
        seq_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )

    vocab = build_vocab(seq_train)

    train_dataset = MedicalDataset(seq_train, y_train, vocab, max_len)
    val_dataset = MedicalDataset(seq_val, y_val, vocab, max_len)
    test_dataset = MedicalDataset(seq_test, y_test, vocab, max_len)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
        vocab
    )


def load_dataset_json(json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    return (
        data["subject_ids"],
        data["hadm_ids"],
        data["sequences"],
        data["labels"]
    )