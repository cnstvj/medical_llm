"""
PyTorch Dataset, vocabulary builder, data loader, and custom train-test splitting utilities.
"""

import json
import random
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from medical_llm.config import DEFAULT_CONFIG


class MedicalDataset(Dataset):
    """PyTorch Dataset for tokenized clinical sequence inputs."""

    def __init__(self, sequences: List[List[str]], labels: List[int], vocab: Dict[str, int], max_len: int = 512):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]

        seq_ids = [self.vocab.get(token, self.vocab["[UNK]"]) for token in seq]

        # Keep MOST RECENT events instead of oldest
        if len(seq_ids) > self.max_len - 1:
            seq_ids = seq_ids[-(self.max_len - 1):]

        seq_ids = [self.vocab["[CLS]"]] + seq_ids

        pad_len = self.max_len - len(seq_ids)
        mask = [1] * len(seq_ids) + [0] * pad_len
        seq_ids += [self.vocab["[PAD]"]] * pad_len

        return (
            torch.tensor(seq_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def train_test_split_custom(
    seqs: List[List[str]], labels: List[int], test_size: float = 0.2, random_state: int = 42
) -> Tuple[List[List[str]], List[List[str]], List[int], List[int]]:
    """Performs stratified train-test split guaranteeing positive & negative classes in both splits."""
    random.seed(random_state)

    class_0 = [(s, l) for s, l in zip(seqs, labels) if l == 0]
    class_1 = [(s, l) for s, l in zip(seqs, labels) if l == 1]

    random.shuffle(class_0)
    random.shuffle(class_1)

    split_0 = int(len(class_0) * (1 - test_size))
    split_1 = max(1, int(len(class_1) * (1 - test_size)))  # at least 1 in train

    train = class_0[:split_0] + class_1[:split_1]
    test = class_0[split_0:] + class_1[split_1:]

    # Ensure test has at least 1 from each class
    if not any(l == 1 for _, l in test) and len(class_1) > 1:
        test.append(train.pop([i for i, (_, l) in enumerate(train) if l == 1][-1]))
    if not any(l == 0 for _, l in test) and len(class_0) > 1:
        test.append(train.pop([i for i, (_, l) in enumerate(train) if l == 0][-1]))

    random.shuffle(train)
    random.shuffle(test)

    seq_train, y_train = zip(*train)
    seq_test, y_test = zip(*test)

    return list(seq_train), list(seq_test), list(y_train), list(y_test)


def build_vocab(train_sequences: List[List[str]]) -> Dict[str, int]:
    """Constructs vocabulary dictionary mapping token strings to integer indices based ONLY on train sequences."""
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[REPORT_START]": 4}
    idx = len(vocab)

    for seq in train_sequences:
        for token in seq:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def get_dataloaders(
    json_path: str = DEFAULT_CONFIG.sequence_dataset_file,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    max_len: int = DEFAULT_CONFIG.max_len,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Loads dataset JSON, splits into train/val/test, builds vocabulary, and returns PyTorch DataLoaders."""
    with open(json_path, "r") as f:
        data = json.load(f)

    sequences = data["sequences"]
    labels = data["labels"]

    seq_train, seq_temp, y_train, y_temp = train_test_split_custom(
        sequences, labels, test_size=0.2, random_state=42
    )

    seq_val, seq_test, y_val, y_test = train_test_split_custom(
        seq_temp, y_temp, test_size=0.5, random_state=42
    )

    # Build vocab ONLY from training data
    vocab = build_vocab(seq_train)

    train_dataset = MedicalDataset(seq_train, y_train, vocab, max_len)
    val_dataset = MedicalDataset(seq_val, y_val, vocab, max_len)
    test_dataset = MedicalDataset(seq_test, y_test, vocab, max_len)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
        vocab,
    )


def load_dataset_json(json_path: str = DEFAULT_CONFIG.sequence_dataset_file) -> Tuple[List[List[str]], List[int]]:
    """Helper function to load sequences and labels from JSON dataset file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["sequences"], data["labels"]
