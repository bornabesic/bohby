
import sklearn, sklearn.datasets, sklearn.neighbors, sklearn.model_selection, sklearn.metrics
import numpy as np
import pickle
import logging

from bohby import optimize_hyperparameters

logging.basicConfig(level = logging.ERROR, format = "%(asctime)s %(message)s",  datefmt = "%H:%M:%S")

# Create a toy dataset here
X, y = sklearn.datasets.make_blobs(n_samples = 1024 * 3, centers = 5, random_state = 0)

# Rake random  train/test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 2/3)

# Random  train/validation split
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size = 1/5)	

# Training & validation function
def train_and_validate(classifier, budget):

    n_train_total = y_train.shape[0]
    indices = np.random.choice(n_train_total, size=int(budget * n_train_total), replace = False)
    
    classifier.fit(X_train[indices], y_train[indices])
    val_score = sklearn.metrics.accuracy_score(y_val, classifier.predict(X_val))
    val_loss = 1 - val_score
    return val_loss

# Optimize using BOHB
valid_loss, config = optimize_hyperparameters(
    sklearn.neighbors.KNeighborsClassifier,
    {
        "n_neighbors": {
            "lower": 1,
            "upper": 20,
            "type": int
        },
        "p": {
            "lower": 1,
            "upper": 6,
            "type": int
        },
        "weights": {
            "choices": ["uniform", "distance"],
            "type": list
        }
    },
    train_and_validate,
    num_iterations = 10
)

# Check final configuration
classifier = sklearn.neighbors.KNeighborsClassifier(**config)
classifier.fit(X_train, y_train)
test_acc = sklearn.metrics.accuracy_score(y_test, classifier.predict(X_test))
print("Best config {}:".format(config))
print("Test loss: {}".format(1 - test_acc))
