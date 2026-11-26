"""
PyTorch Dataset, vocabulary builder, data loader, and patient-level train-test splitting utilities.
Splits are performed at the subject_id level to prevent patient leakage across train/val/test.
"""

import json
import random
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from medical_llm.config import DEFAULT_CONFIG


def _truncate_preserving_demographics_and_recent(
    seq: List[str],
    max_content_len: int,
    d_times: Optional[List[float]] = None,
) -> Tuple[List[str], Optional[List[float]]]:
    """
    Truncates a sequence to `max_content_len` while preserving:
    1. Admission demographic prefix (up to and including '[SEP]' if present near the start).
    2. The most recent (tail) lab trajectory events so acute pre-outcome deterioration is retained.
    """
    if len(seq) <= max_content_len:
        return seq, d_times

    sep_idx = -1
    search_limit = min(10, len(seq))
    for i in range(search_limit):
        if seq[i] == "[SEP]":
            sep_idx = i
            break

    if sep_idx != -1 and (sep_idx + 1) < max_content_len:
        prefix_len = sep_idx + 1
        tail_budget = max_content_len - prefix_len
        new_seq = seq[:prefix_len] + seq[-tail_budget:]
        if d_times is not None:
            padded_dt = list(d_times[:len(seq)])
            if len(padded_dt) < len(seq):
                padded_dt += [0.0] * (len(seq) - len(padded_dt))
            new_dt = padded_dt[:prefix_len] + padded_dt[-tail_budget:]
            return new_seq, new_dt
        return new_seq, None

    # No [SEP] prefix found: keep the most recent tokens
    new_seq = seq[-max_content_len:]
    if d_times is not None:
        padded_dt = list(d_times[:len(seq)])
        if len(padded_dt) < len(seq):
            padded_dt += [0.0] * (len(seq) - len(padded_dt))
        return new_seq, padded_dt[-max_content_len:]
    return new_seq, None


def _resolve_token_id(
    token: str,
    vocab: Dict[str, int],
    unk_dropout: float = 0.0,
) -> int:
    """Maps token to vocab ID with optional training-time UNK dropout and clinical backoff."""
    unk_id = vocab.get("[UNK]", 1)
    if unk_dropout > 0.0 and token not in ("[PAD]", "[CLS]", "[SEP]", "[REPORT_START]", "[UNK]"):
        if random.random() < unk_dropout:
            return unk_id

    if token in vocab:
        return vocab[token]

    if token.startswith("LAB_"):
        if "ABNORMAL" in token.upper() and "LAB_UNKNOWN_ABNORMAL" in vocab:
            return vocab["LAB_UNKNOWN_ABNORMAL"]
        if "LAB_UNKNOWN_NORMAL" in vocab:
            return vocab["LAB_UNKNOWN_NORMAL"]

    return unk_id


class MedicalDataset(Dataset):
    """PyTorch Dataset for tokenized clinical sequence inputs."""

    def __init__(
        self,
        sequences: List[List[str]],
        labels: List[int],
        vocab: Dict[str, int],
        max_len: int = 512,
        unk_dropout: float = 0.0,
    ):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.unk_dropout = unk_dropout

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Preserve demographics and most recent lab trajectory
        seq_trunc, _ = _truncate_preserving_demographics_and_recent(seq, self.max_len - 1)
        seq_ids = [_resolve_token_id(token, self.vocab, unk_dropout=self.unk_dropout) for token in seq_trunc]

        seq_ids = [self.vocab["[CLS]"]] + seq_ids

        pad_len = self.max_len - len(seq_ids)
        mask = [1] * len(seq_ids) + [0] * pad_len
        seq_ids += [self.vocab["[PAD]"]] * pad_len

        return (
            torch.tensor(seq_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


class MultiModalMedicalDataset(Dataset):
    """PyTorch Dataset yielding structured time-series tokens, continuous delta-times, and clinical note tokens."""

    def __init__(
        self,
        lab_sequences: List[List[str]],
        clinical_texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        tokenizer: Any = None,
        max_lab_len: int = 512,
        max_text_len: int = 256,
        delta_times: Optional[List[List[float]]] = None,
        unk_dropout: float = 0.0,
    ):
        self.lab_sequences = lab_sequences
        self.clinical_texts = clinical_texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_lab_len = max_lab_len
        self.max_text_len = max_text_len
        self.delta_times = delta_times
        self.unk_dropout = unk_dropout

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. Structured lab sequence
        seq = self.lab_sequences[idx]
        d_times = self.delta_times[idx] if self.delta_times is not None and idx < len(self.delta_times) else None

        seq_trunc, d_times_trunc = _truncate_preserving_demographics_and_recent(
            seq, self.max_lab_len - 1, d_times=d_times
        )
        seq_ids = [_resolve_token_id(token, self.vocab, unk_dropout=self.unk_dropout) for token in seq_trunc]

        cls_id = self.vocab.get("[CLS]", 2)
        pad_id = self.vocab.get("[PAD]", 0)

        seq_ids = [cls_id] + seq_ids
        if d_times_trunc is not None:
            curr_len = len(seq_ids) - 1
            sliced_d_times = list(d_times_trunc[:curr_len])
            if len(sliced_d_times) < curr_len:
                sliced_d_times += [0.0] * (curr_len - len(sliced_d_times))
            times_list = [0.0] + sliced_d_times
        else:
            times_list = [0.0] * len(seq_ids)

        pad_len = self.max_lab_len - len(seq_ids)
        lab_mask = [1] * len(seq_ids) + [0] * pad_len
        seq_ids += [pad_id] * pad_len
        times_list += [0.0] * pad_len

        # 2. Unstructured clinical notes
        raw_text = str(self.clinical_texts[idx]) if idx < len(self.clinical_texts) and self.clinical_texts[idx] else ""

        if self.tokenizer is not None:
            encoded_text = self.tokenizer(
                raw_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            text_ids = encoded_text["input_ids"].squeeze(0)
            text_mask = encoded_text["attention_mask"].squeeze(0)
        else:
            # Fallback if tokenizer is offline or mock
            text_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            text_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        label = self.labels[idx]

        return {
            "lab_ids": torch.tensor(seq_ids, dtype=torch.long),
            "lab_mask": torch.tensor(lab_mask, dtype=torch.long),
            "delta_times": torch.tensor(times_list, dtype=torch.float32),
            "text_ids": text_ids,
            "text_mask": text_mask,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def train_test_split_by_subject(
    seqs: List[List[str]],
    labels: List[int],
    subject_ids: List[int],
    test_size: float = 0.2,
    random_state: int = 42,
    texts: Optional[List[str]] = None,
) -> Union[
    Tuple[List[List[str]], List[List[str]], List[int], List[int], List[int], List[int]],
    Tuple[List[List[str]], List[List[str]], List[int], List[int], List[int], List[int], List[str], List[str]],
]:
    """Performs patient-level stratified train-test split.

    ALL admissions belonging to the same subject_id stay in the same split.
    Stratification is by whether a patient has any positive (mortality) admission.
    Returns:
        If texts is None: (seq_train, seq_test, y_train, y_test, sid_train, sid_test)
        If texts is provided: (seq_train, seq_test, y_train, y_test, sid_train, sid_test, text_train, text_test)
    """
    random.seed(random_state)

    # Build subject-level data structure
    subject_data: Dict[int, Dict[str, Any]] = {}
    for i, sid in enumerate(subject_ids):
        if sid not in subject_data:
            subject_data[sid] = {"indices": [], "has_positive": False}
        subject_data[sid]["indices"].append(i)
        if labels[i] == 1:
            subject_data[sid]["has_positive"] = True

    # Stratify at subject level
    pos_subjects = [sid for sid, d in subject_data.items() if d["has_positive"]]
    neg_subjects = [sid for sid, d in subject_data.items() if not d["has_positive"]]

    random.shuffle(pos_subjects)
    random.shuffle(neg_subjects)

    split_pos = max(1, int(len(pos_subjects) * (1 - test_size)))
    split_neg = int(len(neg_subjects) * (1 - test_size))

    train_subject_set = set(pos_subjects[:split_pos] + neg_subjects[:split_neg])
    test_subject_set = set(pos_subjects[split_pos:] + neg_subjects[split_neg:])

    # Ensure test has at least 1 positive and 1 negative subject
    if not any(subject_data[s]["has_positive"] for s in test_subject_set) and len(pos_subjects) > 1:
        # Move last positive subject from train to test
        move_sid = pos_subjects[split_pos - 1]
        train_subject_set.discard(move_sid)
        test_subject_set.add(move_sid)

    if all(subject_data[s]["has_positive"] for s in test_subject_set) and len(neg_subjects) > 1:
        # Move last negative subject from train to test
        move_sid = neg_subjects[split_neg - 1]
        train_subject_set.discard(move_sid)
        test_subject_set.add(move_sid)

    # ASSERT: zero patient overlap
    overlap = train_subject_set & test_subject_set
    assert len(overlap) == 0, f"Patient leakage detected! Overlapping subject_ids: {overlap}"

    # Gather indices
    train_indices = [i for sid in train_subject_set for i in subject_data[sid]["indices"]]
    test_indices = [i for sid in test_subject_set for i in subject_data[sid]["indices"]]

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    seq_train = [seqs[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    sid_train = [subject_ids[i] for i in train_indices]

    seq_test = [seqs[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]
    sid_test = [subject_ids[i] for i in test_indices]

    print(f"  Patient-level split: "
          f"train={len(seq_train)} adm ({len(train_subject_set)} patients), "
          f"test={len(seq_test)} adm ({len(test_subject_set)} patients), "
          f"overlap={len(overlap)} patients")

    if texts is not None:
        text_train = [texts[i] for i in train_indices]
        text_test = [texts[i] for i in test_indices]
        return seq_train, seq_test, y_train, y_test, sid_train, sid_test, text_train, text_test

    return seq_train, seq_test, y_train, y_test, sid_train, sid_test



# Keep old name as alias for backward compatibility in imports
def train_test_split_custom(
    seqs: List[List[str]], labels: List[int], test_size: float = 0.2, random_state: int = 42,
    subject_ids: Optional[List[int]] = None,
) -> Tuple[List[List[str]], List[List[str]], List[int], List[int]]:
    """Backward-compatible wrapper. Uses patient-level split when subject_ids provided, else row-level."""
    if subject_ids is not None:
        s_train, s_test, y_train, y_test, _, _ = train_test_split_by_subject(
            seqs, labels, subject_ids, test_size=test_size, random_state=random_state
        )
        return s_train, s_test, y_train, y_test

    # Legacy row-level fallback (only if no subject_ids available)
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
    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[REPORT_START]": 4,
        "LAB_UNKNOWN_NORMAL": 5,
        "LAB_UNKNOWN_ABNORMAL": 6,
    }
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
    """Loads dataset JSON, splits into train/val/test BY PATIENT, builds vocabulary, and returns PyTorch DataLoaders."""
    with open(json_path, "r") as f:
        data = json.load(f)

    sequences = data["sequences"]
    labels = data["labels"]
    subject_ids = data.get("subject_ids")

    if subject_ids is not None:
        # Patient-level split: train/temp (80/20), then temp into val/test (50/50)
        seq_train, seq_temp, y_train, y_temp, sid_train, sid_temp = train_test_split_by_subject(
            sequences, labels, subject_ids, test_size=0.2, random_state=42
        )

        seq_val, seq_test, y_val, y_test, sid_val, sid_test = train_test_split_by_subject(
            seq_temp, y_temp, sid_temp, test_size=0.5, random_state=42
        )

        # Final assertion: no patient overlap across any pair of splits
        train_sids = set(sid_train)
        val_sids = set(sid_val)
        test_sids = set(sid_test)

        assert len(train_sids & val_sids) == 0, f"Train/Val patient overlap: {train_sids & val_sids}"
        assert len(train_sids & test_sids) == 0, f"Train/Test patient overlap: {train_sids & test_sids}"
        assert len(val_sids & test_sids) == 0, f"Val/Test patient overlap: {val_sids & test_sids}"

        print(f"  Final 3-way split (0 patient overlap):")
        print(f"    Train: {len(seq_train)} adm ({len(train_sids)} patients)")
        print(f"    Val  : {len(seq_val)} adm ({len(val_sids)} patients)")
        print(f"    Test : {len(seq_test)} adm ({len(test_sids)} patients)")
    else:
        # Legacy fallback for old JSON files without subject_ids
        print("  WARNING: subject_ids not found in JSON, falling back to row-level split")
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


def load_dataset_json(
    json_path: str = DEFAULT_CONFIG.sequence_dataset_file,
) -> Tuple[List[List[str]], List[int], Optional[List[int]]]:
    """Helper function to load sequences, labels, and subject_ids from JSON dataset file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["sequences"], data["labels"], data.get("subject_ids")
