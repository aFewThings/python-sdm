'''
evaluation on custom dataset.
1) generate pseudo-absence samples based on environmental rasters
2) train SDMs with presence-only(PO) and pseudo-absence samples
3) evaluate them with presence-absence(PA) samples
'''

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import statsmodels.api as sm
from statsmodels.api import GLM # GLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA (Linear Discriminant Analysis) or FDA (Fisher Discriminant Analysis)
from sklearn.neural_network import MLPClassifier # ANN
from sklearn.tree import DecisionTreeClassifier # CTA
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.ensemble import GradientBoostingClassifier # GBM
from xgboost import XGBClassifier # XGB
from lightgbm import LGBMClassifier # LGBM
from pygam import LogisticGAM # GAM
import elapid
from elapid import MaxentModel # MaxEnt
from elapid import NicheEnvelopeModel # SRE

from lib.dataset import EnvironmentalDataset
from lib.raster import PatchExtractor
from lib.raster_metadata import raster_metadata
from lib.evaluation import evaluate
from lib.metrics import ValidationMetricsForBinaryClassification
from lib.utils import make_labels, set_reproducibility

# For reproducibility
random_seed = 42
set_reproducibility(random_seed=random_seed)

# SETTINGS
# files

DATASET_PATH = './data/GBIF_Lithobates_catesbeianus.csv'
RASTER_PATH = './data/rasters_KR/'

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

# sdm models
MODEL_LIST = ['SRE', 'GLM', 'GAM', 'ANN', 'RF', 'GBM', 'XGB', 'LGBM', 'MaxEnt']

SAVE_MODEL_DIR = None
if SAVE_MODEL_DIR:
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

# evaluation
METRICS = (ValidationMetricsForBinaryClassification(verbose=True),)

USE_VIF = False

def remove_multicollinearity(X_train, X_test):
    tempX_train = np.copy(X_train)
    tempX_test = np.copy(X_test)
    while True:
        vif = [variance_inflation_factor(tempX_train, j) for j in range(tempX_train.shape[1])]
        if np.max(vif) < 10 or len(vif) <= 2:
            break
        max_j = np.argmax(vif)
        tempX_train = np.delete(tempX_train, max_j, axis=1)
        tempX_test = np.delete(tempX_test, max_j, axis=1)
    print(f'removed multicollinearity results by VIF: {vif}')
    return tempX_train, tempX_test


if __name__ == '__main__':
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

    train_p_pos, test_p_pos, train_p_labels, test_p_labels \
        = train_test_split(p_pos, p_labels, test_size=TEST_SIZE, train_size=TRAIN_SIZE, random_state=random_seed)

    # To train presence/absence model, we samples pseudo-absence points from valid positions
    # Valid positions are determined by a raster (study area) and presence positions
    train_pa_pos = extractor.get_valid_positions(raster_name='bioclim_1', invalid_pos=train_p_pos, buffer_pos=train_p_pos,
                                                sample_size=8000, drop_nodata=True,
                                                exclusion_dist=EXCLUSION_DIST, local_crs=LOCAL_CRS)
    
    # # NOTE: sre-based pseudo-absence sampling
    # train_p_X_samples = np.stack([extractor[(lat, long)] for (lat, long) in train_p_pos], axis=0)
    # train_pa_X_samples = np.stack([extractor[(lat, long)] for (lat, long) in train_pa_pos], axis=0)
    # # 존재 지역의 환경 변수들을 기준으로 70% 이상 적합한 지역은 의사 부재 위치로 적합하지 않음
    # nem = NicheEnvelopeModel(percentile_range=[2.5, 97.5], overlay='average')
    # nem.fit(train_p_X_samples, train_p_labels, categorical=None)
    # in_range = nem.predict(train_pa_X_samples) > 0.7
    # print('"in_range == False" count: ', sum(in_range == False))
    # train_pa_pos = train_pa_pos[in_range == False, :]

    # under sampling to balance presence/absence samples
    train_pa_pos = train_pa_pos[:len(train_p_pos)]

    ex_pos = np.concatenate((train_p_pos, train_pa_pos, test_p_pos), axis=0)
    bf_pos = np.concatenate((train_p_pos, test_p_pos), axis=0)
    test_pa_pos = extractor.get_valid_positions(raster_name='bioclim_1', invalid_pos=ex_pos, buffer_pos=bf_pos,
                                                sample_size=8000, drop_nodata=True,
                                                exclusion_dist=EXCLUSION_DIST2, local_crs=LOCAL_CRS)
    
    # # NOTE: sre-based pseudo-absence sampling
    # test_pa_X_samples = np.stack([extractor[(lat, long)] for (lat, long) in test_pa_pos], axis=0)
    # in_range = nem.predict(test_pa_X_samples) > 0.7
    # print('"in_range == False" count: ', sum(in_range == False))
    # test_pa_pos = test_pa_pos[in_range == False, :]

    # under sampling to balance presence/absence samples
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

    X_train, y_train = train_set.numpy()
    X_test, y_test = test_set.numpy()

    # Remove Multicollinearity
    if USE_VIF:
        X_train, X_test = remove_multicollinearity(X_train, X_test)

    for model_name in MODEL_LIST:
        if model_name == 'GLM':
            print(f'Training {model_name}...')
            model = GLM(y_train, X_train, family=sm.families.Binomial())
            model_results = model.fit()

            print('Test: ')
            predictions = model_results.predict(X_test)
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                model_results.save(SAVE_MODEL_DIR + "sdm_glm.pkl")

        elif model_name == 'GAM':
            print(f'Training {model_name}...')
            # model = LogisticGAM().fit(X_train, y_train)
            model = LogisticGAM().gridsearch(X_train, y_train)

            print('Test: ')
            predictions = model.predict_proba(X_test)
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                model_results.save(SAVE_MODEL_DIR + "sdm_gam.pkl")

        elif model_name == 'LDA':
            print(f'Training {model_name}...')
            model = LinearDiscriminantAnalysis(n_components=1, solver="svd", store_covariance=True)
            model.fit(X_train, y_train)

            print('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_lda.pkl')

        elif model_name == 'ANN':
            print(f'Training {model_name}...')
            # model = MLPClassifier(hidden_layer_sizes=(100), solver='adam')
            model = MLPClassifier(hidden_layer_sizes=(100), solver='adam', 
                                  verbose=True, learning_rate_init=0.001, 
                                  n_iter_no_change=10, max_iter=200, random_state=random_seed) # default adam, l2 panelty
            model.fit(X_train, y_train)

            print('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_ann.pkl')

        elif model_name == 'CTA':
            print(f'Training {model_name}...')
            model = DecisionTreeClassifier(random_state=random_seed)
            model.fit(X_train, y_train)

            print('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_cta.pkl')

        elif model_name == 'RF':
            print(f'Training {model_name}...')
            model = RandomForestClassifier(n_estimators=100, max_depth=17, n_jobs=16, random_state=random_seed)
            
            model.fit(X_train, y_train)

            print('Test: ')
            predictions = model.predict_proba(X_test)[:, 1] # Nx2(probs of absences, probs of presences)
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_rf.pkl')

        elif model_name == 'GBM':
            print(f'Training {model_name}...')
            model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, random_state=random_seed)

            model.fit(X_train, y_train)

            print('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_gbm.pkl')

        elif model_name == 'XGB':
            print(f'Training {model_name}...')
            #model = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=10, use_label_encoder=False)
            model = XGBClassifier(eval_metric='logloss', random_state=random_seed)

            model.fit(X_train, y_train)

            print('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                model.save_model(SAVE_MODEL_DIR + 'sdm_xgb.pkl')
        
        elif model_name == 'LGBM':
            print(f'Training {model_name}...')
            model = LGBMClassifier(random_state=random_seed, max_depth=4)

            model.fit(X_train, y_train)

            print('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_lgbm.pkl')

        elif model_name == 'MaxEnt':
            print(f'Training {model_name}...')
            # model = MaxentModel(
            #     feature_types = ['linear', 'hinge', 'product'], # the feature transformations
            #     tau = 0.5, # prevalence scaler
            #     clamp = True, # set covariate min/max based on range of training data
            #     scorer = 'roc_auc', # metric to optimize (from sklearn.metrics.SCORERS)
            #     beta_multiplier = 1.0, # regularization scaler (high values drop more features)
            #     beta_lqp = 1.0, # linear, quadratic, product regularization scaler
            #     beta_hinge = 1.0, # hinge regularization scaler
            #     beta_threshold = 1.0, # threshold regularization scaler
            #     beta_categorical = 1.0, # categorical regularization scaler
            #     n_hinge_features = 10, # number of hinge features to compute
            #     n_threshold_features = 10, # number of threshold features to compute
            #     convergence_tolerance = 1e-07, # model fit convergence threshold
            #     use_lambdas = 'best', # set to 'best' (least overfit), 'last' (highest score)
            #     n_cpus = 4, # number of cpu cores to use
            # )

            model = MaxentModel()

            # MaxEnt uses background samples
            b_pos = extractor.get_valid_positions(raster_name='bioclim_1', sample_size=10000)
            b_labels = make_labels(len(b_pos), is_presence=False)
            
            merged_pos = np.concatenate((train_p_pos, b_pos), axis=0)
            merged_labels = np.concatenate((train_p_labels, b_labels), axis=0)
            merged_ids = np.arange(len(merged_pos))

            maxent_train_set = EnvironmentalDataset(merged_labels, merged_pos, merged_ids, patch_extractor=extractor)
            maxent_X_train, maxent_y_train = maxent_train_set.numpy()

            model.fit(maxent_X_train, maxent_y_train)

            print('Test: ')
            predictions = model.predict(X_test)
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'Saving {model_name}...\n')
                elapid.save_object(model, SAVE_MODEL_DIR + 'sdm_maxent.ela')

        elif model_name == 'SRE':
            print(f'Training {model_name}...')
            model = NicheEnvelopeModel(percentile_range=[2.5, 97.5], overlay='intersection')
            
            model.fit(X_train, y_train)

            print('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            print(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                print(f'No Saving option for SRE...\n')

        elif model_name == 'Ensemble':
            print(f'Training {model_name}...')

            models = []
            models.append(RandomForestClassifier(random_state=random_seed))
            models.append(GradientBoostingClassifier(random_state=random_seed))
            models.append(XGBClassifier(eval_metric='logloss', random_state=random_seed))
            models.append(LGBMClassifier(random_state=random_seed))
            models.append(MLPClassifier(hidden_layer_sizes=(100, 100), solver='adam'))

            outputs = []
            for mdl in models:
                mdl.fit(X_train, y_train)
                predictions = mdl.predict_proba(X_test)[:, 1] # Nx2(probs of absences, probs of presences)
                outputs.append(predictions)
            outputs = np.mean(outputs, axis=0)
            print(evaluate(outputs, y_test, METRICS, final=True))

        else:
            print('Wrong model name.')