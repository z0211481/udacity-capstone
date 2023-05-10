from sklearn.neighbors import KNeighborsClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run

from azureml.core import Workspace, Dataset


def clean_data(data):
    x_df = pd.read_csv(data).dropna()
    y_df = x_df.pop("diagnosis").apply(lambda s: 1 if s == "M" else 0)
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n', type=int, default=5,
                        help="Number of neighbors")
    parser.add_argument('--weights', type=str,
                        help="Weight function used for prediction")
    parser.add_argument('--p', type=int, default=2,
                        help="Power parameter for the Minkowski metric")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Num neighbors:", np.int(args.n))
    run.log("Weight function:", args.weights)
    run.log("Metric power:", np.int(args.p))

 
    x, y = clean_data('breast-cancer.csv')

    # TODO: Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=args.n, weights=args.weights, p=args.p).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('./outputs/model', exist_ok=True)

    # save model weights
    joblib.dump(model, './outputs/model/model.h5')

if __name__ == '__main__':
    main()
