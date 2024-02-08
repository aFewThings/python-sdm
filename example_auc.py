import torch
import random
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from lib.dataset import EnvironmentalDataset
from lib.utils import load_model_state
from lib.raster import PatchExtractor
from lib.cnn.models.inception_env import InceptionEnv
from lib.cnn.predict import predict
from lib.evaluation import evaluate
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple


# SETTINGS
N_LABELS = 4520

predictions_cnn = np.load("cnn_test_predictions_logits.npy")
labels_cnn = np.load("cnn_test_labels_logits.npy")
predictions_rf = np.load("test_rf_predictions.npy")
labels_rf = np.load("test_rf_labels.npy")


def selectPseudoAbsence(sp, nb_presence, predictions, labels, selection_type, weight=None):
    if selection_type == "sp":
        # select species for pseudo-absence
        others = []
        pseudo_abscence_predictions = []
        while len(others) < nb_presence:
            # need to have at least as pseudo-absence as presence
            other_sp = sp
            while other_sp == sp:
                # select a random species
                other_sp = random.randint(0, N_LABELS)
            others.extend(labels[labels == other_sp])
            pseudo_abscence_predictions.extend(predictions[labels == other_sp])
        others = np.asarray(others)
        pseudo_abscence_predictions = np.asarray(pseudo_abscence_predictions)
        rand = np.random.choice(others.shape[0], size=nb_presence, replace=False)
        pseudo_abscence_predictions = pseudo_abscence_predictions[rand][:, sp]
        pseudo_abscence = others[rand]
    elif selection_type == "w":
        others = labels[labels != sp]
        pseudo_abscence_predictions = predictions[labels != sp]
        proba = weight[labels != sp] / np.sum(weight[labels != sp])
        rand = np.random.choice(others.shape[0], size=nb_presence, replace=False, p=proba)
        pseudo_abscence_predictions = pseudo_abscence_predictions[rand][:, sp]
        pseudo_abscence = others[rand]
    else:
        # randomly select pseudo-absence
        others = labels[labels != sp]
        pseudo_abscence_predictions = predictions[labels != sp]
        rand = np.random.choice(others.shape[0], size=nb_presence, replace=False)
        pseudo_abscence_predictions = pseudo_abscence_predictions[rand][:, sp]
        pseudo_abscence = others[rand]
    return pseudo_abscence_predictions, pseudo_abscence


def AUCBySpecies(predictions, labels, selection_type):
    list_score = []
    list_tts = []
    compte = 0
    nb_fig = 0

    unique, count = np.unique(labels, return_counts=True)
    weight = np.asarray([1 / count[np.argwhere(unique == l)[0, 0]] for l in labels])

    for SP in range(N_LABELS):
        presence = labels[labels == SP]
        if 1 <= presence.shape[0]:
            # species in the test set
            compte += 1
            presence_predictions = predictions[labels == SP][:, SP]

            pseudo_abscence_predictions, pseudo_abscence = selectPseudoAbsence(SP, presence.shape[0], predictions, labels, selection_type, weight=weight)

            points = np.concatenate((presence, pseudo_abscence))
            points_predictions = np.concatenate((presence_predictions, pseudo_abscence_predictions))

            fpr, tpr, thresholds = metrics.roc_curve(points, points_predictions, pos_label=SP)

            """
            if presence.shape[0] == 1:
                plt.plot(fpr, tpr)
                plt.ylabel('true positive rate')
                plt.xlabel('false positive rate')
                plt.savefig("roc_curve_"+str(nb_fig)+".png")
                nb_fig += 1
                plt.clf()
            """
            score = metrics.auc(fpr, tpr)
            list_score.append(score)
            list_tts.append(score)

    return list_score


def AUCAll(predictions, labels, selection_type):
    points = []
    points_predictions = []

    unique, count = np.unique(labels, return_counts=True)
    weight = np.asarray([1 / count[np.argwhere(unique == l)[0, 0]] for l in labels])

    for SP in range(N_LABELS):
        presence = labels[labels == SP]
        if 1 <= presence.shape[0]:
            presence_predictions = predictions[labels == SP][:, SP]

            pseudo_abscence_predictions, _ = selectPseudoAbsence(SP, presence.shape[0], predictions, labels, selection_type, weight=weight)

            presence_one = [1] * presence.shape[0]
            pseudo_abscence = [0] * presence.shape[0]

            points.extend(presence_one)
            points.extend(pseudo_abscence)
            points_predictions.extend(presence_predictions.tolist())
            points_predictions.extend(pseudo_abscence_predictions.tolist())

    fpr, tpr, thresholds = metrics.roc_curve(points, points_predictions)
    score = metrics.auc(fpr, tpr)

    return score


def AUCMulti(predictions, labels):
    # transform to keep only species in the test set
    predictions = predictions[:, np.unique(labels)]
    predictions = predictions/predictions.sum(axis=1, keepdims=1)
    return metrics.roc_auc_score(labels, predictions, multi_class='ovo', labels=np.unique(labels))


print("-------------- AUC Multi --------------")
#score_nn = AUCMulti(predictions, labels)
#print("cnn:", score_nn)
#score_rf = AUCMulti(predictions_rf, labels_rf)
#print("rf:", score_rf)


SELECT = "w"

print("--------------- AUC all ---------------")
score_nn = AUCAll(predictions_cnn, labels_cnn, SELECT)
score_rf = AUCAll(predictions_rf, labels_rf, SELECT)
print("cnn:", score_nn)
print("rf:", score_rf)
print("--------------- AUC spe ---------------")
list_score_rf = np.sort(AUCBySpecies(predictions_rf, labels_rf, SELECT))
list_score_nn = np.sort(AUCBySpecies(predictions_cnn, labels_cnn, SELECT))
print("cnn:", np.mean(list_score_nn))
print("rf:", np.mean(list_score_rf))

plt.boxplot([list_score_nn, list_score_rf])
plt.ylabel('auc')
plt.savefig('auc_box.png')
plt.clf()
plt.plot(list_score_nn)
plt.plot(list_score_rf)
plt.ylabel('auc')
plt.savefig('auc_curve.png')
