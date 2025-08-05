# ECG Classification Using LSTM Neural Networks

## 🧠 Project Overview

This project presents a deep learning pipeline using Bidirectional LSTM neural networks for the classification of ECG signals. It leverages the MIT-BIH Arrhythmia and PTB Diagnostic datasets to detect and categorize different types of heartbeats, contributing to the early diagnosis of cardiovascular diseases.

The objective is to perform heartbeat classification and improve model accuracy through optimized preprocessing, model design, and evaluation techniques. The project explores the potential of sequential deep learning models in medical signal analysis and aims to serve as a foundation for real-world, AI-driven health monitoring systems.

---

## 📁 Project Structure

```
ecg-lstm-deep-learning/
│
├── data/                         # Contains CSV ECG datasets (not included in repo)
├── notebooks/
│   └── ECG_LSTM.ipynb            # Jupyter Notebook for exploration and result visualization
├── results/
│   ├── metrics/
│   │   ├── accuracy.txt
│   │   └── classification_report.txt
│   └── plots/
│       ├── beat_class_distribution_piechart.png
│       ├── sample_ecg_signal_1.png
│       └── sample_ecg_signals_grid.png
├── src/
│   ├── __init__.py
│   ├── model.py                 # Builds the LSTM model
│   ├── preprocessing.py        # Data loading and preprocessing
│   ├── train.py                # Model training script
│   └── evaluate.py             # Evaluation script (accuracy, report, confusion matrix)
└── main.py                     # Main pipeline to train and evaluate model
```

## 📊 Dataset

This model uses publicly available datasets:

- **MIT-BIH Arrhythmia Dataset**
- **PTB Diagnostic ECG Dataset**

> ❗ **Note:** Due to size limits, the datasets are not included in this repo. Please download them manually and place them in the `data/` folder.

---

## 📦 Requirements

- Python 3.10+
- TensorFlow
- scikit-learn
- pandas
- matplotlib
- seaborn
- numpy

You can install all dependencies via:
pip install -r requirements.txt


🚀 How to Run
1. Clone the repository

git clone https://github.com/JeraldDavidRaj/ecg-lstm-deep-learning.git
cd ecg-lstm-deep-learning

2. Add the datasets
Place the following files inside the data/ folder:

mitbih_train.csv

mitbih_test.csv

ptbdb_normal.csv

ptbdb_abnormal.csv

3. Run the training pipeline

python main.py
This will train the LSTM model and save the final model and metrics in the results/ folder.

## 📈 Results

- ✅ Test accuracy saved in `results/metrics/accuracy.txt`
- 📊 Classification report in `results/metrics/classification_report.txt`
- 📉 Sample ECG signals and pie chart in `results/plots/`

## 🧠 Model Architecture

- Bidirectional LSTM (64 units)
- Dropout (0.2)
- Bidirectional LSTM (32 units)
- Dense layers
- Softmax output layer

Softmax Output

📌 Future Improvements
Hyperparameter tuning

Real-time ECG streaming support

Experiment with 1D CNNs

🤝 Contribution

Contributions are welcome! Please feel free to fork this repo and submit a pull request.

📜 License

This project is licensed under the MIT License.

## 📬 Contact

Jerald David Raj  
📧 [jerald7318@gmail.com](mailto:jerald7318@gmail.com)  
🌐 [GitHub: @JeraldDavidRaj](https://github.com/JeraldDavidRaj)







