import torch as th
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model import Model
from utils import train_epoch, val_epoch


# 0.94 dummy classifier


def main():
    epochs = 1

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                         ])

    train_dataset = ImageFolder(root='data/train', transform=train_transforms)
    val_dataset = ImageFolder(root='data/val', transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

    model = Model(1)
    model.to(device)

    loss = th.nn.BCELoss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(train_loader, model, loss, optimizer, device)
        val_loss = val_epoch(val_loader, model, loss, device)

        print(f'Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

    th.save(model.state_dict(), 'checkpoints/model.pth')


if __name__ == '__main__':
    main()
