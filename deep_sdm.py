'''
evaluation on custom dataset.
1) generate pseudo-absence samples based on environmental rasters
2) train SDMs with presence-only(PO) and pseudo-absence samples
3) evaluate them with presence-absence(PA) samples
'''


import os
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from lib.dataset import EnvironmentalDataset
from lib.raster import PatchExtractor
from lib.raster_metadata import raster_metadata
from lib.cnn.models.dnn import *
from lib.evaluation import evaluate
from lib.metrics import ValidationMetricsForBinaryClassification
from lib.utils import make_labels, set_reproducibility

# For reproducibility
random_seed = 42
set_reproducibility(random_seed=random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RASTER_PATH = './data/rasters_KR/'

DATASET_PATH = './data/GBIF_Lithobates_catesbeianus.csv'

# exclusion buffer
EXCLUSION_DIST = 10000 # exclusion distance
EXCLUSION_DIST2 = 10000
LOCAL_CRS = 5181 # KR

# csv columns
#ID = 'id'
#LABEL = 'Label'
LATITUDE = 'decimalLatitude'
LONGITUDE = 'decimalLongitude'

# dataset construction
TEST_SIZE = 0.3
TRAIN_SIZE = 0.7 # integer or None

# environmental patches
PATCH_SIZE = 1

# training
ITERATIONS = [150] # scheduler iters

# evaluation
METRICS = (ValidationMetricsForBinaryClassification(verbose=True),)


class CustomPrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Start train callback.")
    def on_train_end(self, trainer, pl_module):
        print("End train callback.")
        print("Best Epoch: ", pl_module.best_epoch)
        print(pl_module.best_result)


# Lightning Module: defines whole training, validation, testing process
class DeepSDM(pl.LightningModule):
    def __init__(self, model, criterion, iterations=(30, 50, 100), n_data_dims=39,
                        init_lr=1e-4, gamma=0.1, weight_decay=1e-4,
                        metrics=(ValidationMetricsForBinaryClassification(verbose=True),)):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.iterations = iterations
        self.lr = init_lr
        self.gamma = gamma
        self.weight_decay = weight_decay

        self.metrics = metrics
        self.best_score, self.best_epoch, self.best_result = 0, 0, None

        self.example_input_array = torch.Tensor(1, n_data_dims)
        self.save_hyperparameters(ignore=['model', 'criterion'])

    def configure_optimizers(self):
        # opt = optimizer.SGD(model.parameters(), lr=lr, momentum=momentum)
        opt = optimizer.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = MultiStepLR(opt, milestones=list(self.iterations), gamma=self.gamma)
        return [opt], [scheduler]

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        step_outputs = {}

        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        step_outputs['loss'] = loss
        return step_outputs

    # def training_step_end(self, step_outputs):
    #     raise NotImplementedError()

    # def training_epoch_end(self, total_step_outputs):
    #     raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        step_outputs = {}

        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        step_outputs['outputs'] = outputs
        step_outputs['labels'] = labels
        step_outputs['loss'] = loss
        return step_outputs

    # def validation_step_end(self, step_outputs):
    #     raise NotImplementedError()

    def validation_epoch_end(self, total_step_outputs):
        outputs = []
        labels = []
        #loss = []
        for step_outputs in total_step_outputs:
            outputs.extend(step_outputs['outputs'].data.tolist())
            labels.extend(step_outputs['labels'].data.tolist())
            #loss.extend(step_outputs['loss'])

        result = self._evaluation(outputs, labels)

        if self.metrics[0].auc > self.best_score:
            self.best_score = self.metrics[0].auc
            self.best_epoch = self.current_epoch
            self.best_result = result

        self.log('Metrics/ACC', self.metrics[0].acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Metrics/AUC', self.metrics[0].auc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Metrics/TSS', self.metrics[0].tss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_idx):
        step_outputs = {}

        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        step_outputs['outputs'] = outputs
        step_outputs['labels'] = labels
        step_outputs['loss'] = loss
        return step_outputs

    # def test_step_end(self, step_outputs):
    #     raise NotImplementedError()

    def test_epoch_end(self, total_step_outputs):
        outputs = []
        labels = []
        #loss = []
        for step_outputs in total_step_outputs:
            outputs.extend(step_outputs['outputs'].data.tolist())
            labels.extend(step_outputs['labels'].data.tolist())
            #loss.extend(step_outputs['loss'])

        result = self._evaluation(outputs, labels)

    def _evaluation(self, predictions, labels):
        predictions, labels = np.asarray(predictions), np.asarray(labels)
        result = evaluate(predictions, labels, self.metrics)
        print('\n'*3 + result)
        return result


def main(args):
    # create patch extractor and add all default rasters
    extractor = PatchExtractor(RASTER_PATH, raster_metadata=raster_metadata['default'], size=PATCH_SIZE, verbose=True)
    extractor.add_all(normalized=True, transform=None, ignore=[])

    # READ DATASET
    df = pd.read_csv(DATASET_PATH, header='infer', sep=',', low_memory=False)

    # presence positions
    p_pos = df[[LATITUDE, LONGITUDE]].to_numpy()

    # remove redundant data
    p_pos = extractor.remove_redundant_positions(raster_name='globcover', pos=p_pos)

    # presence labels
    p_labels = make_labels(len(p_pos), is_presence=True)

    # splitting train and testset
    train_p_pos, test_p_pos, train_p_labels, test_p_labels \
        = train_test_split(p_pos, p_labels, test_size=TEST_SIZE, train_size=TRAIN_SIZE, random_state=random_seed)

    # To train presence/absence model, samples pseudo-absence points from valid positions
    # Valid positions are determined by a raster (study area) and presence positions
    train_pa_pos = extractor.get_valid_positions(raster_name='bioclim_1', invalid_pos=train_p_pos, buffer_pos=train_p_pos,
                                                sample_size=8000, drop_nodata=True,
                                                exclusion_dist=EXCLUSION_DIST, local_crs=LOCAL_CRS)
    # under sampling to balance presence/absence samples
    train_pa_pos = train_pa_pos[:len(train_p_pos)]

    ex_pos = np.concatenate((train_p_pos, train_pa_pos, test_p_pos), axis=0) # NOTE: test는 train의 presence, absence 위치를 제외해야함
    bf_pos = np.concatenate((train_p_pos, test_p_pos), axis=0)
    test_pa_pos = extractor.get_valid_positions(raster_name='bioclim_1', invalid_pos=ex_pos, buffer_pos=bf_pos,
                                                sample_size=8000, drop_nodata=True,
                                                exclusion_dist=EXCLUSION_DIST2, local_crs=LOCAL_CRS)
    test_pa_pos = test_pa_pos[:len(test_p_pos)]

    # pseudo-absence pos, labels
    train_pa_pos = train_pa_pos
    train_pa_labels = make_labels(len(train_pa_pos), is_presence=False)
    test_pa_pos = test_pa_pos
    test_pa_labels = make_labels(len(test_pa_pos), is_presence=False)

    # merge presences and pseudo-absences
    train_pos = np.concatenate((train_p_pos, train_pa_pos), axis=0)
    train_labels = np.concatenate((train_p_labels, train_pa_labels), axis=0)
    train_ids = np.arange(len(train_pos))

    test_pos = np.concatenate((test_p_pos, test_pa_pos), axis=0)
    test_labels = np.concatenate((test_p_labels, test_pa_labels), axis=0)
    test_ids = np.arange(len(test_pos))

    # constructing pytorch dataset
    train_set = EnvironmentalDataset(train_labels, train_pos, train_ids, patch_extractor=extractor)
    test_set = EnvironmentalDataset(test_labels, test_pos, test_ids, patch_extractor=extractor)

    # print sampled dataset
    print('train_set presences : ', len(train_set.labels[train_set.labels == 1]))
    print('train_set pseudo-absences : ', len(train_set.labels[train_set.labels == 0]))

    print('test_set presences : ', len(test_set.labels[test_set.labels == 1]))
    print('test_set pseudo-absences : ', len(test_set.labels[test_set.labels == 0]))

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=0)

    # CONSTRUCT MODEL
    classifier = SDM_DNN(in_features=extractor.n_data_dims, 
                          n_labels=args.n_labels, 
                          drop_out=args.dropout, 
                          activation_func=nn.ReLU)

    criterion = nn.BCEWithLogitsLoss()

    model = DeepSDM(model=classifier, criterion=criterion, iterations=ITERATIONS, n_data_dims=extractor.n_data_dims,
                        init_lr=args.init_lr, gamma=args.gamma, weight_decay=args.weight_decay, metrics=METRICS)
    
    logger = TensorBoardLogger('./logs/', name=args.log_dir_name)

    ckpt_callback = ModelCheckpoint(monitor='val_loss',
                                          #dirpath='./logs/',
                                          #every_n_train_steps=0,
                                          #every_n_epochs=0,
                                          filename=args.model_name+'-epoch{epoch:02d}-val_loss{val_loss:.3f}',
                                          auto_insert_metric_name=False)
    print_callback = CustomPrintCallback()
    summary_callback = ModelSummary(max_depth=-1)
    earlystop_callback = EarlyStopping('val_loss', patience=args.patience)

    # defines trainer with CLI args
    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         enable_checkpointing=True,
                                         #resume_from_checkpoint="some/path/to/my_checkpoint.ckpt",
                                         enable_model_summary=True,
                                         num_sanity_val_steps=0,
                                         callbacks=[ckpt_callback, print_callback, earlystop_callback])

    # Model train, validation, test
    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    # logger options
    parser.add_argument('--model_name', type=str, default='litsdm_dnn')
    parser.add_argument('--log_dir_name', type=str, default='test', help='tensorboard log dir')
    
    # trainer
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)

    # dataloader
    parser.add_argument('--batch_size', type=int, default=1)

    # lightning module
    parser.add_argument('--init_lr', type=int, default=1e-3)
    parser.add_argument('--gamma', type=int, default=0.1)
    parser.add_argument('--weight_decay', type=int, default=1e-4)
    
    # model params
    parser.add_argument('--n_labels', type=int, default=1, help='dims of model output')
    parser.add_argument('--dropout', type=int, default=0)

    main(parser.parse_args())