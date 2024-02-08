'''
evaluation on benchmark dataset.
1) generate pseudo-absence samples based on environmental rasters
2) train SDMs with presence-only(PO) and pseudo-absence samples
3) evaluate them with presence-absence(PA) samples
'''

import os
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from statsmodels.api import GLM # GLM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA
from sklearn.neural_network import MLPClassifier # ANN
from sklearn.tree import DecisionTreeClassifier # CTA
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.ensemble import GradientBoostingClassifier # GBM
from xgboost import XGBClassifier # XGB
from lightgbm import LGBMClassifier # LGBM
#from pygam import LogisticGAM # GAM
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
RASTER_PATH = './data/Benchmarks/SWI'
PO_DATASET_PATH = './data/Benchmarks/SWI/SWItrain_po.csv'
PA_DATASET_PATH = './data/Benchmarks/SWI/SWItest_pa.csv'
SPID = 'swi03' # swi01 ~ 30
BASE_RASTER = 'bcc' # base raster for sampling pseudo-absence or background samples

# exclusion buffer
EXCLUSION_DIST = 10000 # exclusion distance
# SWI
RASTER_CRS = 21781
LOCAL_CRS = 21781

LATITUDE = 'y'
LONGITUDE = 'x'

# dataset construction
# TEST_SIZE = 0.2
# TRAIN_SIZE = 0.8 # integer or None

# environmental patches
PATCH_SIZE = 1 # fix

# sdm models
MODEL_LIST = ['GLM', 'LDA', 'ANN', 'RF', 'GBM', 'XGB', 'LGBM', 'MaxEnt', 'SRE']

# SAVE_MODEL_DIR = None
SAVE_MODEL_DIR = f'./pretrained/benchmarks/{SPID}/'

if SAVE_MODEL_DIR:
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    logger.add(SAVE_MODEL_DIR + 'eval.log')
    logger.info("Ready to save logs.")
    logger.info(RASTER_PATH)
    logger.info(PO_DATASET_PATH)
    logger.info(PA_DATASET_PATH)
    logger.info(SPID)
    logger.info(EXCLUSION_DIST)
    logger.info(RASTER_CRS)
    logger.info(LOCAL_CRS)

# evaluation
METRICS = (ValidationMetricsForBinaryClassification(verbose=True),)


if __name__ == '__main__':
    # create patch extractor and add all default rasters
    extractor = PatchExtractor(RASTER_PATH, raster_metadata=raster_metadata['SWI'], size=PATCH_SIZE, verbose=True)
    extractor.add_all(normalized=True, transform=None, ignore=[])

    # NOTE: READ DATASET; training set (PO data)
    df_po = pd.read_csv(PO_DATASET_PATH, header='infer', sep=',', low_memory=False)

    # NOTE: 종 선택
    df_po = df_po[df_po['spid'] == SPID]

    # presence positions
    p_pos = df_po[[LATITUDE, LONGITUDE]].to_numpy()

    # remove redundant data
    # p_pos = extractor.remove_redundant_positions(raster_name=BASE_RASTER, pos=p_pos)

    # presence labels
    p_labels = make_labels(len(p_pos), is_presence=True)

    train_p_pos = p_pos
    train_p_labels = p_labels

    # train_p_pos, test_p_pos, train_p_labels, test_p_labels \
    #     = train_test_split(p_pos, p_labels, test_size=TEST_SIZE, train_size=TRAIN_SIZE, random_state=random_seed)

    # To train presence/absence model, sampling pseudo-absence points from valid positions
    # Valid positions are determined by a raster (study area) and presence positions
    train_pa_pos = extractor.get_valid_positions(raster_name=BASE_RASTER, invalid_pos=train_p_pos, buffer_pos=train_p_pos,
                                                 sample_size=8000, drop_nodata=True,
                                                 exclusion_dist=EXCLUSION_DIST, raster_crs=RASTER_CRS, local_crs=LOCAL_CRS)
    
    # under sampling to balance presence/absence samples
    train_pa_pos = train_pa_pos[:len(train_p_pos)]

    # pseudo-absence pos, labels
    train_pa_pos = train_pa_pos
    train_pa_labels = make_labels(len(train_pa_pos), is_presence=False)

    # NOTE: READ DATASET; test set (PA data)
    df_pa = pd.read_csv(PA_DATASET_PATH, header='infer', sep=',', low_memory=False)
    df_pa = df_pa[[SPID, LATITUDE, LONGITUDE]]
    
    test_p_pos = df_pa[df_pa[SPID] == 1][[LATITUDE, LONGITUDE]].to_numpy()
    test_p_labels = make_labels(len(test_p_pos), is_presence=True)

    test_a_pos = df_pa[df_pa[SPID] == 0][[LATITUDE, LONGITUDE]].to_numpy()
    test_a_labels = make_labels(len(test_a_pos), is_presence=False)

    # merge presences and pseudo-absences
    train_pos = np.concatenate((train_p_pos, train_pa_pos), axis=0)
    train_labels = np.concatenate((train_p_labels, train_pa_labels), axis=0)
    train_ids = np.arange(len(train_pos))

    test_pos = np.concatenate((test_p_pos, test_a_pos), axis=0)
    test_labels = np.concatenate((test_p_labels, test_a_labels), axis=0)
    test_ids = np.arange(len(test_pos))

    # constructing pytorch dataset
    train_set = EnvironmentalDataset(train_labels, train_pos, train_ids, patch_extractor=extractor)
    test_set = EnvironmentalDataset(test_labels, test_pos, test_ids, patch_extractor=extractor)

    # print sampled dataset
    logger.info(f'train_set presences : {len(train_set.labels[train_set.labels == 1])}')
    logger.info(f'train_set pseudo-absences : {len(train_set.labels[train_set.labels == 0])}')

    logger.info(f'test_set presences : {len(test_set.labels[test_set.labels == 1])}')
    logger.info(f'test_set pseudo-absences : {len(test_set.labels[test_set.labels == 0])}')

    X_train, y_train = train_set.numpy()
    X_test, y_test = test_set.numpy()

    for model_name in MODEL_LIST:
        if model_name == 'GLM':
            logger.info(f'Training {model_name}...')
            model = GLM(y_train, X_train)
            model_results = model.fit()

            logger.info('Test: ')
            predictions = model_results.predict(X_test)
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'Saving {model_name}...\n')
                model_results.save(SAVE_MODEL_DIR + "sdm_glm.pkl")

        elif model_name == 'LDA':
            logger.info(f'Training {model_name}...')
            model = LinearDiscriminantAnalysis(n_components=1, solver="svd", store_covariance=True)
            model.fit(X_train, y_train)

            logger.info('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_lda.pkl')

        elif model_name == 'ANN':
            logger.info(f'Training {model_name}...')
            model = MLPClassifier(hidden_layer_sizes=(100), solver='adam')
            model.fit(X_train, y_train)

            logger.info('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_ann.pkl')

        elif model_name == 'CTA':
            logger.info(f'Training {model_name}...')
            model = DecisionTreeClassifier(random_state=random_seed) # NOTE: max_depth를 설정해주지 않으면 cta는 1 또는 0을 출력함.
            model.fit(X_train, y_train)

            logger.info('Test: ')
            predictions = model.predict_proba(X_test)[:, 1] # NOTE: 1 또는 0을 출력하면 ROC 커브를 계산하는데 부적합
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_cta.pkl')

        elif model_name == 'RF':
            logger.info(f'Training {model_name}...')
            model = RandomForestClassifier(n_estimators=100, max_depth=17, n_jobs=16, random_state=random_seed)
            model.fit(X_train, y_train)

            logger.info('Test: ')
            predictions = model.predict_proba(X_test)[:, 1] # Nx2(probs of absences, probs of presences)
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_rf.pkl')

        elif model_name == 'GBM':
            logger.info(f'Training {model_name}...')
            model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, random_state=random_seed)
            model.fit(X_train, y_train)

            logger.info('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_gbm.pkl')

        elif model_name == 'XGB':
            logger.info(f'Training {model_name}...')
            #model = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=10, use_label_encoder=False)
            model = XGBClassifier(eval_metric='logloss', random_state=random_seed)
            model.fit(X_train, y_train)

            logger.info('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'Saving {model_name}...\n')
                model.save_model(SAVE_MODEL_DIR + 'sdm_xgb.pkl')
        
        elif model_name == 'LGBM':
            logger.info(f'Training {model_name}...')
            model = LGBMClassifier(random_state=random_seed, max_depth=4)
            model.fit(X_train, y_train)

            logger.info('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'Saving {model_name}...\n')
                joblib.dump(model, SAVE_MODEL_DIR + 'sdm_lgbm.pkl')

        elif model_name == 'MaxEnt':
            logger.info(f'Training {model_name}...')
            model = MaxentModel(
                feature_types = ['linear', 'hinge', 'product'], # the feature transformations
                tau = 0.5, # prevalence scaler
                clamp = True, # set covariate min/max based on range of training data
                scorer = 'roc_auc', # metric to optimize (from sklearn.metrics.SCORERS)
                beta_multiplier = 1.0, # regularization scaler (high values drop more features)
                beta_lqp = 1.0, # linear, quadratic, product regularization scaler
                beta_hinge = 1.0, # hinge regularization scaler
                beta_threshold = 1.0, # threshold regularization scaler
                beta_categorical = 1.0, # categorical regularization scaler
                n_hinge_features = 10, # number of hinge features to compute
                n_threshold_features = 10, # number of threshold features to compute
                convergence_tolerance = 1e-07, # model fit convergence threshold
                use_lambdas = 'best', # set to 'best' (least overfit), 'last' (highest score)
                n_cpus = 4, # number of cpu cores to use
            )

            # MaxEnt uses background samples
            b_pos = extractor.get_valid_positions(raster_name=BASE_RASTER, sample_size=10000)
            b_labels = make_labels(len(b_pos), is_presence=False)
            
            merged_pos = np.concatenate((train_p_pos, b_pos), axis=0)
            merged_labels = np.concatenate((train_p_labels, b_labels), axis=0)
            merged_ids = np.arange(len(merged_pos))

            maxent_train_set = EnvironmentalDataset(merged_labels, merged_pos, merged_ids, patch_extractor=extractor)
            maxent_X_train, maxent_y_train = maxent_train_set.numpy()

            model.fit(maxent_X_train, maxent_y_train)

            logger.info('Test: ')
            predictions = model.predict(X_test)
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'Saving {model_name}...\n')
                elapid.save_object(model, SAVE_MODEL_DIR + 'sdm_maxent.ela')

        elif model_name == 'SRE':
            logger.info(f'Training {model_name}...')
            model = NicheEnvelopeModel(percentile_range=[2.5, 97.5], overlay='intersection')
            
            model.fit(X_train, y_train)

            logger.info('Test: ')
            predictions = model.predict_proba(X_test)[:, 1]
            logger.info(evaluate(predictions, y_test, METRICS, final=True))

            if SAVE_MODEL_DIR:
                logger.info(f'No Saving option for SRE...\n')

        elif model_name == 'Ensemble':
            logger.info(f'Training {model_name}...')

            models = []
            models.append(RandomForestClassifier(random_state=random_seed))
            models.append(GradientBoostingClassifier(random_state=random_seed))
            models.append(XGBClassifier(eval_metric='logloss', random_state=random_seed))
            models.append(LGBMClassifier(random_state=random_seed))
            models.append(MLPClassifier(hidden_layer_sizes=(100), solver='adam'))

            outputs = []
            for mdl in models:
                mdl.fit(X_train, y_train)
                predictions = mdl.predict_proba(X_test)[:, 1] # Nx2(probs of absences, probs of presences)
                outputs.append(predictions)
            outputs = np.mean(outputs, axis=0)
            logger.info(evaluate(outputs, y_test, METRICS, final=True))

        else:
            logger.info('Wrong model name.')