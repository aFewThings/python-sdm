import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lib.dataset import EnvironmentalDataset
from lib.raster import PatchExtractor
from lib.evaluation import evaluate
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple

# This is an example code how to use a scikit-learn model with our dataset (here a random forest classifier)

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
MAX_DEPTH = 17
N_TREES = 100

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


# CONSTRUCT MODEL
clf = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, n_jobs=16)


# TRAINING
print('Training...')
X, y = train_set.numpy()
clf.fit(X, y)

# save model
path = 'rf.skl'
print('Saving SKL model: ' + path)
joblib.dump(clf, path)


# VALIDATION
print('Validation: ')
inputs, labels = validation_set.numpy()
restricted_predictions = clf.predict_proba(inputs)
# sklearn fit() doesn't take as input the labels or the number of classes. It's infer with the training data.
# Labels order in prediction of the clf model is given in clf.classes_.
# With random split, some species can be only on the test or validation set and will not be present in the prediction.
# So, the prediction need to be reshape to covers all species as following.
predictions = np.zeros((restricted_predictions.shape[0], N_LABELS))
predictions[:, clf.classes_] = restricted_predictions

print(evaluate(predictions, labels, METRICS))


# FINAL EVALUATION ON TEST SET
# read test set
df = pd.read_csv(TESTSET_PATH, header='infer', sep=';', low_memory=False)
ids = df[ID].to_numpy()
labels = df[LABEL].to_numpy()
positions = df[[LATITUDE, LONGITUDE]].to_numpy()
test_set = EnvironmentalDataset(labels, positions, ids, patch_extractor=extractor)

# load model
path = 'rf.skl'
print('Loading SKL model: ' + path)
clf = joblib.load(path)

# predict
inputs, labels = test_set.numpy()
restricted_predictions = clf.predict_proba(inputs)
predictions = np.zeros((restricted_predictions.shape[0], N_LABELS))
predictions[:, clf.classes_] = restricted_predictions

# evaluate
print('Final test:')
print(evaluate(predictions, labels, METRICS, final=True))
