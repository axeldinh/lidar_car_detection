import torch
from load_data import LidarModule
from models import get_model
import pytorch_lightning as pl

class LightningModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = get_model(params)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        loss = self.model.get_loss(data, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        loss = self.model.get_loss(data, label)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

def train(params, debug=False):

    datamodule = LidarModule(params["data"], debug=debug)
    model = LightningModule(params["model"])

    trainer = pl.Trainer(**params["trainer"])
    trainer.fit(model, datamodule)
    #trainer.predict(model, datamodule)


if __name__ == '__main__':
    import yaml
    from time import time

    with open('config.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    debug = params['debug']

    times = []
    num_workers = [0, 1, 2, 3, 4]

    for num_worker in num_workers:
        params['data']['num_workers'] = num_worker
        start = time()
        train(params, debug=debug)
        times.append(time() - start)

    for num_worker, time in zip(num_workers, times):
        print(f"num_workers: {num_worker}, time: {time}")
