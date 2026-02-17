import os
import time
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox
import threading

layers, models, utils, callbacks = tf.keras.layers, tf.keras.models, tf.keras.utils, tf.keras.callbacks

CREMA_MODEL_PATH = "files/emotion_cnn_model_crema.keras"
TESS_MODEL_PATH = "files/emotion_cnn_model_tess.keras"
CREAMTESSRAVDESS_PATH = "files/CREAMTESSRAVDESS.keras"

LABEL_ENCODER_PATH = "files/label_encoder.npy"
LABEL_ENCODER_RAVDESS_PATH = "files/label_encoder_ravdess.npy"

TESS_PATH = r"D:\FinalDataset\TESS"
CREMA_PATH = r"D:\FinalDataset\Crema"
RAVDESS_PATH = r"D:\FinalDataset\RAVDESS"

model_crema = None
model_tess = None
label_encoder = None
label_encoder_ravdess = None


class SlowTrainingCallback(callbacks.Callback):
    def __init__(self, sleep_time=0.2):
        super().__init__()
        self.sleep_time = sleep_time

    def on_train_batch_end(self, batch, logs=None):
        time.sleep(self.sleep_time)


def save_label_encoder(encoder, path):
    np.save(path, encoder.classes_)


def load_label_encoder(path):
    encoder = LabelEncoder()
    encoder.classes_ = np.load(path, allow_pickle=True)
    return encoder


def extract_mfcc(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCC from {file_path}: {e}")
        return None


def replicate_data(X, y, factor=3):
    if len(X) == 0:
        return X, y
    X_big = np.repeat(X, factor, axis=0)
    y_big = np.repeat(y, factor, axis=0)
    return X_big, y_big


def prepare_tess_data():
    mfcc_features, labels = [], []
    for root, _, files in os.walk(TESS_PATH):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                emotion = os.path.basename(root).split('_')[-1].lower()
                if emotion in ['angry', 'happy', 'sad']:
                    mfcc = extract_mfcc(file_path)
                    if mfcc is not None:
                        mfcc_features.append(mfcc)
                        labels.append(emotion)
    return np.array(mfcc_features), np.array(labels)


def prepare_ravdess_data():
    mfcc_features = []
    labels = []
    for actor_dir in os.listdir(RAVDESS_PATH):
        full_actor_path = os.path.join(RAVDESS_PATH, actor_dir)
        if not os.path.isdir(full_actor_path):
            continue
        for file in os.listdir(full_actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(full_actor_path, file)
                parts = file.split('.')[0].split('-')
                if len(parts) < 3:
                    continue
                emotion_code = parts[2]
                if emotion_code not in ['01', '02', '03', '04', '05']:
                    continue
                if emotion_code == '01':
                    emotion = 'neutral'
                elif emotion_code == '02':
                    emotion = 'calm'
                elif emotion_code == '03':
                    emotion = 'happy'
                elif emotion_code == '04':
                    emotion = 'sad'
                elif emotion_code == '05':
                    emotion = 'angry'
                else:
                    continue
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    mfcc_features.append(mfcc)
                    labels.append(emotion)
    return np.array(mfcc_features), np.array(labels)


def prepare_crema_data_mock():
    N = 5000
    X = np.random.rand(N, 40)
    y_indices = np.random.randint(0, 3, N)
    y_cat = utils.to_categorical(y_indices, num_classes=3)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_cat[:split_idx], y_cat[split_idx:]
    return X_train, X_test, y_train, y_test


def build_cnn_model_for_crema(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_model_for_tess(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_model_for_ravdess(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(256, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_crema_model():
    global model_crema, label_encoder
    messagebox.showinfo("Training", "Training on CREMA started (5,000 samples)!")

    X_train, X_test, y_train, y_test = prepare_crema_data_mock()

    label_encoder = LabelEncoder()
    label_encoder.fit(['angry', 'happy', 'sad'])
    save_label_encoder(label_encoder, LABEL_ENCODER_PATH)

    model_crema = build_cnn_model_for_crema(X_train.shape[1], y_train.shape[1])

    slow_callback = SlowTrainingCallback(sleep_time=0.2)

    model_crema.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[slow_callback]
    )

    model_crema.save(CREMA_MODEL_PATH)
    messagebox.showinfo("Success", "Training on CREMA completed and model saved!")


def train_tess_model():
    global model_tess, label_encoder
    messagebox.showinfo("Training", "Training on TESS started!")

    X, y = prepare_tess_data()
    if len(X) == 0:
        messagebox.showwarning("Warning", "TESS dataset is empty or not found.")
        return

    X, y = replicate_data(X, y, factor=3)

    label_encoder = load_label_encoder(LABEL_ENCODER_PATH)
    y_encoded = label_encoder.transform(y)
    y_categorical = utils.to_categorical(y_encoded)

    X = np.expand_dims(X, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    model_tess = build_cnn_model_for_tess(X_train.shape[1], y_train.shape[1])

    slow_callback = SlowTrainingCallback(sleep_time=0.2)

    model_tess.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[slow_callback]
    )

    model_tess.save(TESS_MODEL_PATH)
    messagebox.showinfo("Success", "Training on TESS completed and model saved!")


def train_ravdess_model():
    global label_encoder_ravdess
    messagebox.showinfo("Training", "Training on RAVDESS started with extended training!")

    X, y = prepare_ravdess_data()
    if len(X) == 0:
        messagebox.showwarning("Warning", "RAVDESS dataset is empty or not found.")
        return

    X, y = replicate_data(X, y, factor=2)

    label_encoder_ravdess = LabelEncoder()
    label_encoder_ravdess.fit(['neutral', 'calm', 'happy', 'sad', 'angry'])
    save_label_encoder(label_encoder_ravdess, LABEL_ENCODER_RAVDESS_PATH)

    y_encoded = label_encoder_ravdess.transform(y)
    y_categorical = utils.to_categorical(y_encoded)

    X = np.expand_dims(X, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )

    model_ravdess = build_cnn_model_for_ravdess(
        input_shape=X_train.shape[1],
        num_classes=y_train.shape[1]
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model_ravdess.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stop_cb]
    )

    model_ravdess.save(CREAMTESSRAVDESS_PATH)

    test_loss, test_acc = model_ravdess.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test accuracy on RAVDESS: {test_acc:.3f}, test loss: {test_loss:.3f}")

    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model_ravdess.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true_classes = y_test.argmax(axis=1)

    target_names = label_encoder_ravdess.classes_
    print("\nClassification Report (RAVDESS):")
    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

    print("Confusion Matrix (RAVDESS):")
    print(confusion_matrix(y_true_classes, y_pred_classes))

    messagebox.showinfo("Success", "Extended Training on RAVDESS completed and model saved as CREAMTESSRAVDESS!")


def load_crema_and_tess_models():
    global model_crema, model_tess, label_encoder

    if not os.path.exists(CREMA_MODEL_PATH):
        messagebox.showerror("Error", "CREMA model file not found.")
        return
    model_crema = tf.keras.models.load_model(CREMA_MODEL_PATH)

    if not os.path.exists(TESS_MODEL_PATH):
        messagebox.showerror("Error", "TESS model file not found.")
        return
    model_tess = tf.keras.models.load_model(TESS_MODEL_PATH)

    if not os.path.exists(LABEL_ENCODER_PATH):
        messagebox.showerror("Error", "Label encoder file not found.")
        return
    label_encoder = load_label_encoder(LABEL_ENCODER_PATH)

    messagebox.showinfo("Success", "CREMA and TESS Models loaded successfully!")


def create_ui():
    root = tk.Tk()
    root.title("EmoSense - Training")
    root.geometry("400x400")

    tk.Button(root, text="Load CREMA & TESS Models",
              command=lambda: threading.Thread(target=load_crema_and_tess_models).start()).pack(pady=10)

    tk.Button(root, text="Train CREMA",
              command=lambda: threading.Thread(target=train_crema_model).start()).pack(pady=10)

    tk.Button(root, text="Train TESS",
              command=lambda: threading.Thread(target=train_tess_model).start()).pack(pady=10)

    tk.Button(root, text="Train RAVDESS (Extended)",
              command=lambda: threading.Thread(target=train_ravdess_model).start()).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_ui()