import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Separate features and labels
    x_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    x_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    # Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

