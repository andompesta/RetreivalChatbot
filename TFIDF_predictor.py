import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.IO_data import create_csv_batch_iter, create_row_iter
from utils.eval_metrics import evaluate_recall

BATCH_SIZE=20000


class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(data)

    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]

if __name__ == '__main__':
    # Load Data
    train_df = create_csv_batch_iter('train.csv', path='data', batch_size=BATCH_SIZE)
    test_df = create_csv_batch_iter('test.csv', path='data',batch_size=BATCH_SIZE)
    validation_df = create_csv_batch_iter("valid.csv", path='data', batch_size=BATCH_SIZE)

    # y_test = np.zeros(len(test_df))


    pred = TFIDFPredictor()
    pred.train(create_row_iter(train_df, func=lambda x:" ".join(x[:2])))    # pass all the row of the training file Contex " " Utterance


    y = [pred.predict(row[0], row[1:]) for row in create_row_iter(test_df)]
    for n in [1, 2, 5, 10]:
        print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y, np.zeros(len(y)), n)))
