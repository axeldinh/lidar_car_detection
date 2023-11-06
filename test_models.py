"""Test the models on a simple training loop."""
from load_data import LidarModule
from models import (
    PointNetClassifier,
    PointNetRegressor,
    PointNetPPRegressor,
    PointNetPPClassifier,
)
import torch
from tqdm import tqdm
import cProfile
from pstats import Stats, SortKey


def val_loop(model, val_loader, epoch_idx):
    model.eval()
    total_error = 0
    max_error = 0
    device = next(model.parameters()).device

    val_pb = tqdm(val_loader)

    for batch in val_pb:
        data, label = batch
        data, label = data.to(device), label.to(device)
        loss, preds = model.get_loss(data, label)
        diffs = torch.abs(preds.view(-1) - label.cpu().view(-1))
        total_error += diffs.sum().item()
        max_error = max(max_error, diffs.max().item())

    print(f"Epoch {epoch_idx} - Total Error: {total_error} - Max Error: {max_error}")
    model.train()


def train_model(model, train_loader, val_loader, epochs=5):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in tqdm(range(epochs)):
        for batch in tqdm(train_loader):
            data, label = batch
            data, label = data.to(device), label.to(device)
            loss, _ = model.get_loss(data, label)
            loss.backward()
            optimizer.step()

        val_loop(model, val_loader, epoch)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using ", "GPU 🎉" if device == "cuda" else "CPU 😢")

    datamodule = LidarModule(batch_size=2, debug=True)
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
        train_model(model, train_loader, val_loader)


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

        with open("profile.txt", "w") as f:
            stats = Stats(pr, stream=f)
            stats.strip_dirs()
            stats.sort_stats("cumtime")
            stats.dump_stats(".profile_stats")
            stats.print_stats()
