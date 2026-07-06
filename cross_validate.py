import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sequence_builder import EXPERIMENT_MODE

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from dataset import (
    MedicalDataset,
    build_vocab,
    load_dataset_json
)

from model import MedicalTransformer


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N_SPLITS = 5
MAX_EPOCHS = 30
PATIENCE = 3
BATCH_SIZE = 32


def evaluate(model, loader):

    model.eval()

    probs = []
    labels = []

    with torch.no_grad():

        for x, mask, y in loader:

            x = x.to(DEVICE)
            mask = mask.to(DEVICE)

            logits = model(x, mask)

            p = torch.sigmoid(logits)

            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.96, 0.05):

        preds = [
            1 if p > threshold else 0
            for p in probs
        ]

        f1 = f1_score(
            labels,
            preds,
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    preds = [
        1 if p > best_threshold else 0
        for p in probs
    ]

    precision = precision_score(
        labels,
        preds,
        zero_division=0
    )

    recall = recall_score(
        labels,
        preds,
        zero_division=0
    )

    cm = confusion_matrix(
        labels,
        preds
    )

    return (
        auc,
        best_f1,
        precision,
        recall,
        best_threshold,
        cm
    )


def train_fold(
    train_sequences,
    train_labels,
    test_sequences,
    test_labels,
    fold,
    vocab
):

    train_dataset = MedicalDataset(
        train_sequences,
        train_labels,
        vocab
    )

    test_dataset = MedicalDataset(
        test_sequences,
        test_labels,
        vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    model = MedicalTransformer(
        len(vocab)
    ).to(DEVICE)

    pos = sum(train_labels)
    neg = len(train_labels) - pos

    pos_weight = torch.tensor(
        [neg / max(pos, 1)],
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS
    )

    best_auc = -1
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):

        model.train()

        total_loss = 0

        for x, mask, y in train_loader:

            x = x.to(DEVICE)
            mask = mask.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(x, mask)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        (
            val_auc,
            _,
            _,
            _,
            _,
            _
        ) = evaluate(
            model,
            test_loader
        )

        print(
            f"Fold {fold} "
            f"Epoch {epoch+1} "
            f"Loss={total_loss:.4f} "
            f"Val_AUROC={val_auc:.4f}"
        )

        if val_auc > best_auc:

            best_auc = val_auc

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                f"best_fold_{fold}.pth"
            )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                f"Early stopping "
                f"(Fold {fold})"
            )

            break

    model.load_state_dict(
        torch.load(
            f"best_fold_{fold}.pth",
            map_location=DEVICE
        )
    )

    (
        auc,
        f1,
        precision,
        recall,
        threshold,
        cm
    ) = evaluate(
        model,
        test_loader
    )

    print("\n" + "="*40)
    print(f"FOLD {fold} RESULTS")
    print("="*40)

    print(f"AUROC     : {auc:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Threshold : {threshold:.2f}")

    print("\nConfusion Matrix")
    print(cm)

    return (
        auc,
        f1,
        precision,
        recall
    )


def main():

    sequences, labels = load_dataset_json(
        "sequence_dataset.json"
    )

    outer_train_sequences, outer_val_sequences, outer_train_labels, outer_val_labels = train_test_split(
        sequences,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print(
        f"Total samples={len(sequences)}, "
        f"outer train={len(outer_train_sequences)}, "
        f"outer validation={len(outer_val_sequences)}"
    )

    vocab = build_vocab(outer_train_sequences)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    aucs = []
    f1s = []
    precisions = []
    recalls = []

    best_cv_auc = -1.0
    best_fold = None

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(outer_train_sequences, outer_train_labels),
        start=1
    ):

        train_sequences = [
            outer_train_sequences[i]
            for i in train_idx
        ]

        train_labels = [
            outer_train_labels[i]
            for i in train_idx
        ]

        test_sequences = [
            outer_train_sequences[i]
            for i in test_idx
        ]

        test_labels = [
            outer_train_labels[i]
            for i in test_idx
        ]

        (
            auc,
            f1,
            precision,
            recall
        ) = train_fold(
            train_sequences,
            train_labels,
            test_sequences,
            test_labels,
            fold,
            vocab
        )

        aucs.append(auc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

        if auc > best_cv_auc:
            best_cv_auc = auc
            best_fold = fold

    print("\n")
    print("="*50)
    print("FINAL CROSS-VALIDATION RESULTS - "+EXPERIMENT_MODE)
    print("="*50)

    print(
        f"Mean AUROC     : "
        f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
    )

    print(
        f"Mean F1        : "
        f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
    )

    print(
        f"Mean Precision : "
        f"{np.mean(precisions):.4f}"
    )

    print(
        f"Mean Recall    : "
        f"{np.mean(recalls):.4f}"
    )


if __name__ == "__main__":
    main()