import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # star_1, star_2 correspond to the different star rows
    def __distance(self, star_1, star_2):
        dist = ((star_1[0]-star_2[0])/3)**2 + (star_1[1]-star_2[1])**2
        return dist

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            dist = []
            for idx, ele in enumerate(self.X):
                dist.append((self.__distance(x, ele),self.y[idx]))
            
            dist.sort( key=lambda tup: tup[0])
            classes = [x[1] for x in dist]
            classes = classes[0:self.K]
            
            preds.append(max(set(classes), key=classes.count))
        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y