from typing import Any
import torch
from load_data import LidarModule
from models import get_model
import pytorch_lightning as pl

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
        total_error = torch.abs(preds - label).mean()
        max_error = torch.abs(preds - label).max()
        self.log('train_loss', loss)
        outputs = {
            'total_error': total_error,
            'max_error': max_error
        }
        self.training_steps_outputs.append(outputs)
        return loss
    
    def on_train_epoch_end(self):
        total_error = torch.stack([x['total_error'] for x in self.training_steps_outputs]).mean()
        max_error = torch.stack([x['max_error'] for x in self.training_steps_outputs]).max()
        self.log('train_total_error', total_error)
        self.log('train_max_error', max_error)

    def validation_step(self, batch, batch_idx):
        data, label = batch
        loss, preds = self.model.get_loss(data, label)
        total_error = torch.abs(preds - label).mean()
        max_error = torch.abs(preds - label).max()
        self.log('val_loss', loss)
        outputs = {
            'total_error': total_error,
            'max_error': max_error
        }
        self.validation_steps_outputs.append(outputs)

    def on_validation_epoch_end(self):
        total_error = torch.stack([x['total_error'] for x in self.validation_steps_outputs]).mean()
        max_error = torch.stack([x['max_error'] for x in self.validation_steps_outputs]).max()
        self.log('val_total_error', total_error)
        self.log('val_max_error', max_error)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

def train(params, debug=False):

    datamodule = LidarModule(params["data"], debug=debug)
    model = LightningModule(params["model"])

    trainer = pl.Trainer(**params["trainer"])
    trainer.fit(model, datamodule)
    predictions = trainer.predict(model, datamodule)
    predictions = torch.cat(predictions)
    print(predictions)


if __name__ == '__main__':
    import yaml

    with open('config.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    debug = params['debug']
    if debug:
        params['trainer']['max_epochs'] = 1
        params['trainer']['log_every_n_steps'] = 1
        params['data']['batch_size'] = 2


    train(params, debug)