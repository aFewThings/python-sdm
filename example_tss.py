import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


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

def TSSScore(predictions, labels, selection_type):
    # scaler = RobustScaler(quantile_range=(0.10, 0.90))
    # predictions = scaler.fit_transform(predictions)
    scaler = MinMaxScaler()
    predictions = scaler.fit_transform(predictions)

    list_tpr = [[] for i in range(999)]
    list_tnr = [[] for i in range(999)]

    unique, count = np.unique(labels, return_counts=True)
    weight = np.asarray([1 / count[np.argwhere(unique == l)[0, 0]] for l in labels])

    for SP in range(N_LABELS):
        presence = labels[labels == SP]
        if 1 <= presence.shape[0]:
            presence_predictions = predictions[labels == SP][:, SP]

            pseudo_abscence_predictions, pseudo_abscence = selectPseudoAbsence(SP, presence.shape[0], predictions, labels, selection_type, weight=weight)

            for i in range(999):
                list_tpr[i].append(np.sum(presence_predictions >= ((i + 1) * 0.001)) / presence_predictions.size)
                list_tnr[i].append(np.sum(pseudo_abscence_predictions < ((i + 1) * 0.001)) / pseudo_abscence_predictions.size)

    tpr = np.mean(np.asarray(list_tpr), axis=1)
    tnr = np.mean(np.asarray(list_tnr), axis=1)

    tss = np.add(tpr, tnr)
    tss = tss - 1
    return tss


SELECT = "w"

tss_rf = TSSScore(predictions_rf, labels_rf, SELECT)
tss_cnn = TSSScore(predictions_cnn, labels_cnn, SELECT)

print(np.argmax(tss_cnn))
print(np.argmax(tss_rf))
print(np.max(tss_cnn))
print(np.max(tss_rf))
