
import json
import torch
from torch.utils.data import Dataset, DataLoader
import random

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

        seq_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in seq]

        # Keep MOST RECENT events instead of oldest
        if len(seq_ids) > self.max_len - 1:
            seq_ids = seq_ids[-(self.max_len - 1):]

        seq_ids = [self.vocab['[CLS]']] + seq_ids

        pad_len = self.max_len - len(seq_ids)
        mask = [1] * len(seq_ids) + [0] * pad_len
        seq_ids += [self.vocab['[PAD]']] * pad_len

        return (
            torch.tensor(seq_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )

def train_test_split_custom(seqs, labels, test_size=0.2, random_state=42):
    random.seed(random_state)

    # Stratified split: separate by class, shuffle each, then
    # draw proportionally so both classes appear in every split.
    class_0 = [(s, l) for s, l in zip(seqs, labels) if l == 0]
    class_1 = [(s, l) for s, l in zip(seqs, labels) if l == 1]

    random.shuffle(class_0)
    random.shuffle(class_1)

    split_0 = int(len(class_0) * (1 - test_size))
    split_1 = max(1, int(len(class_1) * (1 - test_size)))  # at least 1 in train

    train = class_0[:split_0] + class_1[:split_1]
    test  = class_0[split_0:] + class_1[split_1:]

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

def build_vocab(train_sequences):
    vocab = {'[PAD]':0,'[CLS]':1,'[SEP]':2,'[UNK]':3,'[REPORT_START]':4}
    idx = len(vocab)

    for seq in train_sequences:
        for token in seq:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def get_dataloaders(json_path, batch_size=32, max_len=512):
    with open(json_path,"r") as f:
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
        DataLoader(train_dataset,batch_size=batch_size,shuffle=True),
        DataLoader(val_dataset,batch_size=batch_size),
        DataLoader(test_dataset,batch_size=batch_size),
        vocab
    )

def load_dataset_json(json_path):
    import json

    with open(json_path, "r") as f:
        data = json.load(f)

    return data["sequences"], data["labels"]