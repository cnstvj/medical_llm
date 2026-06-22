
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import MedicalTransformer
from sklearn.metrics import roc_auc_score, f1_score

def evaluate(model, loader, device):
    model.eval()

    probs = []
    labels = []

    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            logits = model(x, mask)
            p = torch.sigmoid(logits)

            probs.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    preds = [1 if p > 0.5 else 0 for p in probs]

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    return auc, f1_score(labels, preds, zero_division=0)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        "sequence_dataset.json"
    )

    model = MedicalTransformer(len(vocab)).to(device)

    labels = train_loader.dataset.labels
    pos = sum(labels)
    neg = len(labels) - pos

    pos_weight = torch.tensor([neg/max(pos,1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    best_auc = -1

    for epoch in range(10):
        model.train()

        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x, mask)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        val_auc, val_f1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: AUROC={val_auc:.4f} F1={val_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(),"best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))

    test_auc, test_f1 = evaluate(model, test_loader, device)
    print(f"TEST AUROC={test_auc:.4f} F1={test_f1:.4f}")

if __name__ == "__main__":
    train()
