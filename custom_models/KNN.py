import numpy as np

class KNN:


    def calc(self, X_train, y_train, x_test):
        distance = []
        targets = dict()

        for X in range(len(X_train)):
            distance.append((KNN.func_dist(X_train[X], x_test), X))

        distance.sort()

        for i in range(self.k):
            index = distance[i][1]
            if targets.get(y_train[index]) != None:
                targets[y_train[index]] += 1
            else:
                targets[y_train[index]] = 1

        return max(targets, key=targets.get)
        

    def predict(self, X_train, X_test, y_train):
        predictions = []

        for item in X_test:
            predictions.append(self.calc(X_train, y_train, item))

        return predictions

    def set_k_ind(self, num):
        self.k = num

        

    @staticmethod
    def func_dist(x, y):
        return abs(np.sum(x-y))