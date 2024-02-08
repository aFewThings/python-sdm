from abc import ABC
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr


class ValidationMetric(ABC):
    def __init__(self, final_validation=False, sort_needed=False):
        self.final_validation = final_validation
        self.sort_needed = sort_needed

    def get_result(self):
        return self.result

    def __repr__(self):
        return self.__class__.__name__


class ValidationAccuracyMultiple(ValidationMetric):
    def __init__(self, list_top_k=(10,), final_validation=False):
        super().__init__(final_validation, True)
        self.list_top_k = list_top_k
        self.result = np.zeros(len(self.list_top_k))

    def __call__(self, predictions, labels):
        res = np.zeros(len(self.list_top_k), dtype=int)
        for i, pred in enumerate(predictions):
            for j, top_k in enumerate(self.list_top_k):
                answer = pred[0:top_k]
                if labels[i] in answer:
                    res[j] += 1
        self.result = res / labels.shape[0]
        return self.__str__()

    def __str__(self):
        list_out = []
        for i in range(len(self.list_top_k)):
            list_out.append(
                'Top-'+str(self.list_top_k[i]) + ' accuracy of the model on the test set: %.4f' % self.result[i]
            )
        return '\n'.join(list_out)


class ValidationAccuracyMultipleBySpecies(ValidationMetric):
    def __init__(self, list_top_k, final_validation=False):
        super().__init__(final_validation, True)
        self.list_top_k = list_top_k
        self.result = np.zeros(self.list_top_k)

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        result_sp = np.zeros((nb_labels, len(self.list_top_k)))
        count = np.zeros(nb_labels)
        keep = np.zeros(nb_labels, dtype=bool)
        for i, pred in enumerate(predictions):
            rg = np.argwhere(pred == labels[i])[0, 0]
            count[labels[i]] += 1
            keep[labels[i]] = True
            for j, k in enumerate(self.list_top_k):
                if rg <= k:
                    result_sp[labels[i], j] += 1
        count = count[keep]
        count = count[:, np.newaxis]
        result_sp = result_sp[np.array(keep), :]
        result_sp = result_sp/count
        self.result = np.sum(result_sp, 0) / count.shape[0]

        return self.__str__()

    def __str__(self):
        list_out = []
        for i in range(len(self.list_top_k)):
            list_out.append(
                'Top-' + str(
                    self.list_top_k[i]) + ' accuracy by species of the model on the test set: %.4f' % self.result[i]
            )
        return '\n'.join(list_out)


class ValidationAccuracyRange(ValidationMetric):
    def __init__(self, max_top_k=100, final_validation=False):
        super().__init__(final_validation, True)
        self.file_name = "result_range_top"+str(max_top_k)+".npy"
        self.max_top_k = max_top_k
        self.result = np.zeros(self.max_top_k)

    def __call__(self, predictions, labels):
        for i, pred in enumerate(predictions):
            rg = np.argwhere(pred == labels[i])
            for j in range(self.max_top_k):
                if rg <= j:
                    self.result[j] += 1
        self.result = self.result / labels.shape[0]
        np.save(self.file_name, self.result)
        return self.__str__()

    def __str__(self):
        return "Results of accuracy range saved in file \'"+self.file_name+"\'"


class ValidationAccuracyRangeBySpecies(ValidationMetric):
    def __init__(self, max_top_k=100, final_validation=False):
        super().__init__(final_validation, True)
        self.file_name = "result_range_top"+str(max_top_k)+"_by_species.npy"
        self.max_top_k = max_top_k
        self.result = np.zeros(self.max_top_k)

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        result_sp = np.zeros((nb_labels, self.max_top_k))
        count = np.zeros(nb_labels)
        keep = np.zeros(nb_labels, dtype=bool)
        for i, pred in enumerate(predictions):
            rg = np.argwhere(pred == labels[i])[0, 0]
            count[labels[i]] += 1
            keep[labels[i]] = True
            for j in range(self.max_top_k):
                if rg <= j:
                    result_sp[labels[i], j] += 1
        count = count[keep]
        count = count[:, np.newaxis]
        result_sp = result_sp[np.array(keep), :]
        result_sp = result_sp/count
        self.result = np.sum(result_sp, 0) / count.shape[0]
        np.save(self.file_name, self.result)
        return self.__str__()

    def __str__(self):
        return "Results of accuracy range by species saved in file \'"+self.file_name+"\'"


class ValidationAccuracyForAllSpecies(ValidationMetric):
    def __init__(self, train, top_k=30, n_species=4520, final_validation=False):
        super().__init__(final_validation, True)
        self.file_name = "result_top"+str(top_k)+"_for_all_species.npy"
        self.top_k = top_k
        self.train = train
        self.prior = np.zeros(n_species, dtype=int)
        for label in self.train.labels:
            self.prior[label] += 1

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        result_sp = np.zeros(nb_labels)
        count = np.zeros(nb_labels)
        keep = np.zeros(nb_labels, dtype=bool)
        for i, pred in enumerate(predictions):
            rg = np.argwhere(pred == labels[i])[0, 0]
            count[labels[i]] += 1
            keep[labels[i]] = True
            if rg <= self.top_k:
                result_sp[labels[i]] += 1
        self.prior = self.prior[keep]
        count = count[keep]
        result_sp = result_sp[keep]
        result_sp = result_sp/count
        result_sp = result_sp[np.argsort(-self.prior)]
        np.save(self.file_name, result_sp)
        return self.__str__()

    def __str__(self):
        return "Results of accuracy top"+str(self.top_k)+" for all species saved in file \'"+self.file_name+"\'"


class ValidationMRR(ValidationMetric):
    def __init__(self, val_limit=None, final_validation=False):
        super().__init__(final_validation, True)
        self.val_limit = val_limit
        self.result = 0

    def __call__(self, predictions, labels):
        if self.val_limit is not None:
            predictions = predictions[:, :self.val_limit]
        res = 0
        for i, pred in enumerate(predictions):
            pos = np.where(pred == labels[i])
            if pos[0].shape[0] != 0:
                res += 1.0 / (pos[0][0] + 1)
        self.result = res / labels.shape[0]
        return self.__str__()

    def __str__(self):
        return 'MRR of the model on the test set: %.4f' % self.result


class ValidationMRRBySpecies(ValidationMetric):
    def __init__(self, final_validation=False):
        super().__init__(final_validation, True)
        self.result = 0

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        res = np.zeros(nb_labels, dtype=float)
        count = np.zeros(nb_labels, dtype=int)
        keep = np.zeros(nb_labels, dtype=bool)
        for i, pred in enumerate(predictions):
            count[labels[i]] += 1
            keep[labels[i]] = True
            res[labels[i]] += 1.0 / (np.where(pred == labels[i])[0][0] + 1)
        res = res[keep]
        count = count[keep]
        self.result = np.sum(np.divide(res, count), dtype=float) / res.shape[0]
        return self.__str__()

    def __str__(self):
        return 'MSMRR of the model on the test set: %.4f' % self.result


#---#


class ValidationMetricsForBinaryClassification(ValidationMetric):
    def __init__(self, final_validation=False, verbose=True):
        super().__init__(final_validation, True)
        self.verbose = verbose

        self.cutoff = 0.5

        self.acc = np.zeros((1))
        self.auc = np.zeros((1))
        self.kappa = np.zeros((1))
        self.cm = np.zeros((2, 2))
        self.sen = np.zeros((1)) # True Positive Rate
        self.spe = np.zeros((1)) # True Negative Rate
        self.tss = np.zeros((1))
        self.cor = np.zeros((1)) # Pearson correlation coefficient

    def __call__(self, pred_probs, labels):
        pred_probs[pred_probs < 0.0] = 0.0
        pred_probs[pred_probs > 1.0] = 1.0

        # threshold-independent metric
        self.auc = roc_auc_score(labels, pred_probs)
        self.cor, _ = pearsonr(pred_probs, labels)

        # get min and max probs
        mini = max(min(pred_probs), 0.0)
        maxi = min(max(pred_probs), 1.0)
        num_tests = 100
        test_cutoffs = np.linspace(mini, maxi, num=num_tests)
        
        optimal_cutoff = 0
        best_tss = 0
        sen = spe = tss = 0
        for c in test_cutoffs:
            pred_binary = np.where(pred_probs > c, 1.0, 0.0)
            
            cm = confusion_matrix(labels, pred_binary)
            tn, fp, fn, tp = cm.ravel()

            sen = (tp / (tp + fn))
            spe = (tn / (tn + fp))
            tss = sen + spe - 1

            if tss > best_tss: # tss 기준
                best_tss = tss
                optimal_cutoff = c

        # threshold-dependent metrics
        # Maximum TSS을 위한 threshold를 찾고 이를 다른 metric에도 적용
        self.cutoff = optimal_cutoff
        pred_binary = np.where(pred_probs > optimal_cutoff, 1.0, 0.0)
        self.cm = confusion_matrix(labels, pred_binary)
        tn, fp, fn, tp = self.cm.ravel()

        self.sen = (tp / (tp + fn))
        self.spe = (tn / (tn + fp))
        self.tss = self.sen + self.spe - 1
        self.acc = accuracy_score(labels, pred_binary)
        self.kappa = cohen_kappa_score(labels, pred_binary)

        return self.__str__()

    def __str__(self):
        list_out = []
        if self.verbose:
            list_out.append(f'optimized threshold(best TSS): {self.cutoff}')
            list_out.append(f'Accuracy: {self.acc}')
            list_out.append(f'AUC: {self.auc}')
            list_out.append(f'Sensitivity(TPR): {self.sen}')
            list_out.append(f'Specificity(TNR): {self.spe}')
            list_out.append(f'KAPPA: {self.kappa}')
            list_out.append(f'TSS: {self.tss}')
            list_out.append(f'COR: {self.cor}')
            tn, fp, fn, tp = self.cm.ravel()
            list_out.append(f'Confusion Matrix: tn {tn}, fp {fp}, fn {fn}, tp {tp}')
        else:
            list_out.append(f'AUC: {self.auc}')
            list_out.append(f'TSS: {self.tss}')
        return '\n'.join(list_out)