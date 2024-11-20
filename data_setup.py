import os
import platform

import cv2
from torch.utils.data import DataLoader, Dataset

NUM_WORKERS = 0 if platform.system() == "Windows" else os.cpu_count()


class MRI(Dataset):
    def __init__(self, X, y, transform=None):
        self.classes = [
            "glioma_tumor",
            "meningioma_tumor",
            "no_tumor",
            "pituitary_tumor",
        ]
        self.cls_to_idx = {
            "glioma_tumor": 0,
            "meningioma_tumor": 1,
            "no_tumor": 2,
            "pituitary_tumor": 3,
        }
        self.idx_to_cls = {
            0: "glioma_tumor",
            1: "meningioma_tumor",
            2: "no_tumor",
            3: "pituitary_tumor",
        }
        self.X = X
        self.y = [self.cls_to_idx[lbl] for lbl in y]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.X[idx])
        label = self.y[idx]
        if self.transform:
            return self.transform(image=img)["image"], label
        else:
            return img, label


def create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    transform=None,
    batch_size=32,
    num_workers=NUM_WORKERS,
):
    train_data = MRI(X_train, y_train, transform=transform)
    val_data = MRI(X_val, y_val, transform=transform)
    test_data = MRI(X_test, y_test, transform=transform)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, test_dataloader
