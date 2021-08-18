from math import log, exp


class AdaBoost:
    def __init__(self, classifiers, samples, labels):
        self.classifiers = classifiers
        self.h = []
        self.a = []
        self.learn(samples, labels)

    @staticmethod
    def sign(x):
        return +1 if x >= 0 else -1

    def predict(self, sample):
        out = 0
        for classifier_index, weight in zip(self.h, self.a):
            out += weight * self.classifiers[classifier_index](sample)
        return self.sign(out)

    def learn(self, samples, labels):
        k = 0
        k_max = len(self.classifiers)
        weights = [1/len(samples)] * len(samples)

        while k < k_max:
            k += 1

            hk = 0
            ek = float("inf")
            for classifier_i, classifier in enumerate(self.classifiers):
                classifier_cost = 0
                for sample_i, (sample, label) in enumerate(zip(samples, labels)):
                    if self.sign(classifier(sample)) != label:
                        classifier_cost += weights[sample_i]
                if classifier_cost < ek:
                    hk = classifier_i
                    ek = classifier_cost
            del classifier_i, classifier, classifier_cost, sample_i, sample, label

            if ek > 0.5:
                k_max = k_max - 1
                break

            ak = 0.5 * log((1 - ek) / ek)
            wk = []
            for wki, xi, yi in zip(weights, samples, labels):
                wk.append(wki * exp(-1 * ak * yi * self.classifiers[hk](xi)))
            Zk = sum(wk)
            wk = [wk_i / Zk for wk_i in wk]
            weights = wk

            self.h.append(hk)
            self.a.append(ak)

            misclassified = 0
            for sample, label in zip(samples, labels):
                if self.predict(sample) != label:
                    misclassified += 1

            print(" > Classifier with index %i with weight %f, no misclassified %i" % (hk, ak, misclassified))
