import matplotlib.pyplot as plt
import torch
from load_data import LidarModule
from models import get_model
import pytorch_lightning as pl
from utils import make_submission


class LightningModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = get_model(params)

        self.training_steps_outputs = []
        self.validation_steps_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        loss, preds = self.model.get_loss(data, label)
        if not self.params["regression"]:
            preds = torch.argmax(preds, dim=1).float()
        diffs = torch.abs(preds.view(-1) - label)
        self.log("train_loss", loss)
        outputs = {"diffs": diffs}
        self.training_steps_outputs.append(outputs)
        return loss

    def on_train_epoch_end(self):
        diffs = torch.stack([x["diffs"] for x in self.training_steps_outputs]).view(-1)
        total_error = diffs.mean() * 601  # Number of train samples
        max_error = diffs.max()
        self.log("train_total_error", total_error)
        self.log("train_max_error", max_error)
        self.training_steps_outputs = []

    def validation_step(self, batch, batch_idx):
        data, label = batch
        loss, preds = self.model.get_loss(data, label)
        if not self.params["regression"]:
            preds = torch.argmax(preds, dim=1).float()
        diffs = torch.abs(preds.view(-1) - label)
        self.log("val_loss", loss)
        outputs = {
            "diffs": diffs,
        }
        self.validation_steps_outputs.append(outputs)

    def on_validation_epoch_end(self):
        import wandb

        diffs = torch.stack([x["diffs"] for x in self.validation_steps_outputs]).view(
            -1
        )
        total_error = diffs.mean()  # Number of test samples
        max_error = diffs.max()
        self.log("val_total_error", total_error)
        self.log("val_max_error", max_error)
        self.validation_steps_outputs = []

        totals = []
        vals = [0.0]

        while vals[-1] < 12:
            total = (
                torch.logical_and(vals[-1] <= diffs, diffs < vals[-1] + 1.0)
                .sum()
                .item()
            )
            totals.append(total)
            vals.append(vals[-1] + 1.0)

        vals = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
        totals = [x / len(diffs) for x in totals]

        fig, ax = plt.subplots()
        ax.plot(vals, totals)
        ax.set_xlabel("Error")
        ax.set_ylabel("Percentage")
        ax.set_title("Error distribution")

        wandb.log({"error_distribution": fig})

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.params["learning_rate"]
        )


def train(params, debug=False):
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import LearningRateMonitor

    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_error",
        dirpath="checkpoints",
        filename="best",
        save_top_k=1,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_total_error", patience=10, verbose=True, mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    wandb_logger = WandbLogger(
        name=params["model"]["model"],
        project="lidar-car-detection",
        log_model=True,
        offline=False,
    )
    tensorboard_logger = TensorBoardLogger("logs")

    wandb_logger.log_hyperparams(params)
    tensorboard_logger.log_hyperparams(params)

    logger = [wandb_logger, tensorboard_logger]
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    datamodule = LidarModule(params["data"], debug=debug)
    model = LightningModule(params["model"])

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **params["trainer"])
    trainer.fit(model, datamodule)

    predictions = trainer.predict(model, datamodule, ckpt_path="best")
    if predictions is None:
        print("All predictions are None.")
        return
    predictions = [torch.tensor(x) for x in predictions]
    predictions = torch.cat(predictions, dim=0).view(-1).cpu().numpy()

    make_submission(predictions)


if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    debug = params["debug"]
    if debug:
        params["trainer"]["max_epochs"] = 1
        params["trainer"]["log_every_n_steps"] = 1
        params["data"]["batch_size"] = 2

    train(params, debug)
