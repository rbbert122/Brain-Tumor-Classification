from copy import deepcopy

import numpy as np
import torch
from tqdm.auto import tqdm


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(
    model,
    dataloader,
    loss_fn,
    device,
):
    model.eval()

    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            val_pred_logits = model(X)

            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += (val_pred_labels == y).sum().item() / len(val_pred_labels)

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    epochs,
    device,
    verbose=False,
):
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_model_wts = deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in tqdm(range(epochs)):
        current_lr = get_lr(optimizer)
        if verbose:
            print("Epoch {}/{}, current lr={}".format(epoch, epochs - 1, current_lr))

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())

            torch.save(model.state_dict(), "weights.pt")

        scheduler.step(val_loss)
        if current_lr != get_lr(optimizer):
            if verbose:
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        if verbose:
            print(
                f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {100*val_acc:.2f}"
            )
            print("-" * 10)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    model.load_state_dict(best_model_wts)

    return model, results
