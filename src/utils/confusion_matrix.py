import numpy as np


class ConfusionMatrix:
    def __init__(self, classes):
        self.classes = classes
        num_classes = len(classes)
        self.cm = np.zeros(shape=(num_classes, num_classes))
        self.strFormat = '%-10s:%-10s'

    def update(self, pred, target):
        assert len(pred) == len(target)
        assert isinstance(pred, np.ndarray)
        assert isinstance(target, np.ndarray)

        for p, t in zip(pred, target):
            self.cm[p][t] += 1

    def get_accuracy(self):
        return np.round(self.cm.diagonal().sum() / self.cm.sum(), 2)
    
    def get_accuracy_per_cls(self):
        # return np.round(self.cm.diagonal() / self.cm.sum(1), 2)
        return {cls: acc for cls, acc in 
                zip(self.classes, np.round(self.cm.diagonal() / self.cm.sum(1), 2))}

    def print_calc(self):
        print("=" * 20)       
        print("Accuracy")
        print("-" * 20)

        for cls, acc in self.get_accuracy_per_cls().items():
            print(self.strFormat % (cls, acc))
        print("-" * 20)
        print(self.strFormat % ("accuracy", self.get_accuracy()))

        print("=" * 20)

    def clear(self):
        self.__init__(self.classes)