import numpy as np
import pandas as pd

from utils.eval_metrics import evaluate_recall


def create_csv_batch_iter(file_name, path='data', batch_size=20000):
    for df in pd.read_csv(path + '/' + file_name, chunksize=batch_size):
        yield df

# Random Predictor
def predict_random(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)

train_df = create_csv_batch_iter('train.csv')
test_df = create_csv_batch_iter('test.csv')
validation_df = create_csv_batch_iter("valid.csv")


for test_batch in test_df:
    # Evaluate Random predictor
    y_random = [predict_random(test_batch.Context[x], test_batch.iloc[x,1:].values) for x in range(len(test_batch))]
    y_test = np.zeros(len(y_random))
    for n in [1, 2, 5, 10]:
        print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y_random, y_test, n)))
