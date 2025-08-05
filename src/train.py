import os
from src.model import build_model
from src.preprocessing import load_data, preprocess_data

def train():
    train_df, test_df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Save model to top-level results/
    os.makedirs("../results", exist_ok=True)
    model.save("../results/ecg_lstm_model.h5")

    return model, X_test, y_test

