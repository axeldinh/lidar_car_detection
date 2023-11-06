import matplotlib.pyplot as plt
import torch
from load_data import LidarModule
from models import get_model
import pytorch_lightning as pl
from utils import make_submission
import wandb


class LightningModule(pl.LightningModule):
    def __init__(self, model_name, regression, lr=1e-4, model_kwargs={}):
        super().__init__()

        self.model_name = model_name
        self.regression = regression
        self.lr = lr
        self.model = get_model(
            model_name=model_name, regression=regression, model_kwargs=model_kwargs
        )

        self.training_steps_outputs = []
        self.validation_steps_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        loss, preds = self.model.get_loss(data, label)
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
        diffs = torch.abs(preds.view(-1) - label)
        self.log("val_loss", loss)
        outputs = {
            "diffs": diffs,
        }
        self.validation_steps_outputs.append(outputs)

    def on_validation_epoch_end(self):
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

        try:
            wandb.log({"error_distribution": fig})
        except Exception:
            pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def train(params, debug=False):
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import LearningRateMonitor

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="val_total_error",
            dirpath="checkpoints",
            filename="best",
            save_top_k=1,
            mode="min",
        )
    )

    callbacks.append(LearningRateMonitor(logging_interval="step"))

    if params["early_stop"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_total_error", patience=10, verbose=True, mode="min"
            )
        )

    loggers = []
    loggers.append(TensorBoardLogger("logs"))
    if params["wandb"]:
        loggers.append(
            WandbLogger(
                name=params["lit_params"]["model_name"],
                project="lidar-car-detection",
                log_model=True,
                offline=False,
            )
        )
    for logger in loggers:
        logger.log_hyperparams(params)

    datamodule = LidarModule(debug=debug, **params["data_module_kwargs"])
    model = LightningModule(**params["lit_params"], model_kwargs=params["model_kwargs"])

    trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **params["trainer"])
    trainer.fit(model, datamodule)

    predictions = trainer.predict(model, datamodule, ckpt_path="best")
    predictions = torch.cat(predictions, dim=0).view(-1).cpu().numpy()  # type: ignore

    make_submission(predictions)


if __name__ == "__main__":
    import yaml
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", type=str, default="config.yaml")
    args.add_argument("-d", "--debug", action="store_true", default=False)
    args.add_argument("-w", "--wandb", action="store_true", default=False)
    args.add_argument("-e", "--early_stop", action="store_true", default=False)
    args = args.parse_args()

    with open(args.config) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        if params["debug"] is None:
            params["debug"] = args.debug
        else:  # Command line argument overrides config.yaml
            params["debug"] = args.debug or params["debug"]  # This overrides it

    debug = params["debug"]
    if debug:
        params["trainer"]["max_epochs"] = 1
        params["trainer"]["log_every_n_steps"] = 1
        params["data_module_kwargs"]["batch_size"] = 2

    params["wandb"] = args.wandb
    params["early_stop"] = args.early_stop

    train(params, debug)
