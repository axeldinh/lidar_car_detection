"""Test the models on a simple training loop."""
from load_data import LidarModule
from models import (
    PointNetClassifier,
    PointNetRegressor,
    PointNetPPRegressor,
    PointNetPPClassifier,
)
import torch
import cProfile
from pstats import Stats
from tqdm import tqdm


def val_loop(model, val_loader, epoch_idx, verbose=False):
    model.eval()
    total_error = 0
    max_error = 0
    device = next(model.parameters()).device

    if verbose:
        val_pb = tqdm(val_loader)
    else:
        val_pb = val_loader
    for batch in val_pb:
        data, label = batch
        data, label = data.to(device), label.to(device)
        loss, preds = model.get_loss(data, label)
        diffs = torch.abs(preds.view(-1) - label.cpu().view(-1))
        total_error += diffs.sum().item()
        max_error = max(max_error, diffs.max().item())

    if verbose:
        val_pb.set_description(
            f"Validation - Epoch {epoch_idx} - Total Error: {total_error} - Max Error: {max_error}"
        )
    model.train()


def train_model(model, train_loader, val_loader, epochs=2, verbose=False):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        if verbose:
            epoch_pb = tqdm(train_loader)
            epoch_pb.set_description(f"Training - Epoch {epoch}")
        else:
            epoch_pb = train_loader
        for batch in epoch_pb:
            data, label = batch
            data, label = data.to(device), label.to(device)
            loss, _ = model.get_loss(data, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_loop(model, val_loader, epoch, verbose=verbose)


def main(debug=False, epochs=2, batch_size=2, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using ", "GPU 🎉" if device == "cuda" else "CPU 😢")

    datamodule = LidarModule(batch_size=batch_size, debug=debug)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    models = [
        PointNetClassifier,
        PointNetRegressor,
        PointNetPPRegressor,
        PointNetPPClassifier,
    ]

    for model in models:
        print("==========================================")
        print(f"Training {model.__name__}...")
        model = model()
        train_model(model, train_loader, val_loader, epochs=epochs, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with cProfile.Profile() as pr:
        main(
            debug=args.debug,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        with open("profile.txt", "w") as f:
            stats = Stats(pr, stream=f)
            stats.strip_dirs()
            stats.sort_stats("cumtime")
            stats.dump_stats(".profile_stats")
            stats.print_stats()
