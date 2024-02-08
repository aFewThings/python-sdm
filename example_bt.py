import ast
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from lib.dataset import EnvironmentalDataset
from lib.raster import PatchExtractor
from lib.evaluation import evaluate
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple

# This is an example code how to use xgboost models with our dataset
# this code is only working with gpu

# SETTINGS
# files
TRAINSET_PATH = './data/train_dataset.csv'
TESTSET_PATH = './data/test_dataset.csv'
RASTER_PATH = './data/rasters/'
# csv columns
ID = 'id'
LABEL = 'Label'
LATITUDE = 'Latitude'
LONGITUDE = 'Longitude'

# dataset construction
VAL_SIZE = 0.1

# environmental patches
PATCH_SIZE = 1

# model params
N_LABELS = 4520
MAX_DEPTH = 2
N_ROUND = 360

# evaluation
METRICS = (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationAccuracyMultiple([1, 10, 30]))


# READ DATASET
df = pd.read_csv(TRAINSET_PATH, header='infer', sep=';', low_memory=False)

ids = df[ID].to_numpy()
labels = df[LABEL].to_numpy()
positions = df[[LATITUDE, LONGITUDE]].to_numpy()

# splitting train val test
train_labels, val_labels, train_positions, val_positions, train_ids, val_ids\
    = train_test_split(labels, positions, ids, test_size=VAL_SIZE, random_state=42)

# create patch extractor
extractor = PatchExtractor(RASTER_PATH, size=PATCH_SIZE, verbose=True)
# add all default rasters
extractor.add_all()

# constructing pytorch dataset
train_set = EnvironmentalDataset(train_labels, train_positions, train_ids, patch_extractor=extractor)
validation_set = EnvironmentalDataset(val_labels, val_positions, val_ids, patch_extractor=extractor)


# TRAINING
print('Training...')
X, y = train_set.numpy()
d_train = xgb.DMatrix(X, label=y)

inputs, labels = validation_set.numpy()
d_val = xgb.DMatrix(inputs, label=labels)

eval_list = [(d_val, 'eval'), (d_train, 'train')]

# model and learning parameters, for more information read the xgboost documentation
params = {'objective': 'multi:softprob', 'max_depth': MAX_DEPTH, 'seed': 4242, 'eval_metric': 'merror',
          'num_class': N_LABELS, 'early_stopping_rounds': 10, 'updater': 'grow_gpu', 'predictor': 'gpu_predictor',
          'tree_method': 'gpu_hist', 'gpu_id': 0, 'verbosity': 2}

# the model is create with the train
bst = xgb.train(
    params,
    d_train,
    num_boost_round=N_ROUND,
    verbose_eval=1,
    evals=eval_list
)


# save model
path = 'bt'
print("Saving model: " + path)
# save also the best iteration (not saved with the model)
complement = {'best_iteration': bst.best_ntree_limit}
with open(path + "_complement.txt", "w") as file:
    file.write(str(complement))
bst.save_model(path + ".xgb")

# predict and evaluate on validation
predictions = bst.predict(d_val, ntree_limit=bst.best_ntree_limit)
print(evaluate(predictions, labels, METRICS))


# FINAL EVALUATION ON TEST SET
# read test set
df = pd.read_csv(TESTSET_PATH, header='infer', sep=';', low_memory=False)
ids = df[ID].to_numpy()
labels = df[LABEL].to_numpy()
positions = df[[LATITUDE, LONGITUDE]].to_numpy()
test_set = EnvironmentalDataset(labels, positions, ids, patch_extractor=extractor)

# load model
path = 'bt'
print("Loading model: " + path)
bst = xgb.Booster()
bst.load_model(path + ".xgb")
# load the best iterations
with open(path + "_complement.txt", "r") as file:
    st = file.read()
    complement = ast.literal_eval(st)
if 'best_iteration' in complement:
    bst.best_ntree_limit = complement['best_iteration']


print('Final test:')
# predict
inputs, labels = test_set.numpy()
d_val = xgb.DMatrix(inputs, label=labels)

# evaluate
predictions = bst.predict(d_val, ntree_limit=bst.best_ntree_limit)
print(evaluate(predictions, labels, METRICS))
