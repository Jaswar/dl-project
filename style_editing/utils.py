import torch as th
from tqdm import tqdm


def train_epoch(train_loader, model, loss, optimizer, device):
    model.train()
    total_loss = 0
    total_preds = 0
    correct_preds = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        y = y.unsqueeze(1).float()
        y_hat = model(x)

        optimizer.zero_grad()
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()

        total_loss += l.item()
        correct_preds += ((y_hat > 0.5) == y).sum().item()
        total_preds += y.shape[0]

        if i % 10 == 0:
            print(f'Iteration: {i}/{len(train_loader)} - Batch loss: {l.item()} - Accuracy: {correct_preds / total_preds * 100}')

    return total_loss / len(train_loader)


def val_epoch(val_loader, model, loss, device):
    model.eval()
    total_loss = 0
    total_preds = 0
    correct_preds = 0
    with th.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            y = y.unsqueeze(1).float()

            y_hat = model(x)
            l = loss(y_hat, y)

            total_loss += l.item()
            correct_preds += ((y_hat > 0.5) == y).sum().item()
            total_preds += y.shape[0]

            if i % 10 == 0:
                print(f'Iteration: {i}/{len(val_loader)} - Batch loss: {l.item()} - Accuracy: {correct_preds / total_preds * 100}')

    return total_loss / len(val_loader)


