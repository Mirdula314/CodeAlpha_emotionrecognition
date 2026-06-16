import os
import librosa
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = "RAVDESS"

emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

def extract_features(file_path):
    audio, sr = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0)

X = []
y = []

print("Loading Dataset...")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            emotion_code = file.split("-")[2]

            emotion = emotion_map[emotion_code]

            path = os.path.join(root, file)

            features = extract_features(path)

            X.append(features)
            y.append(emotion)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print(f"\nAccuracy: {accuracy:.2%}")

joblib.dump(model, "emotion_model.pkl")

print("Model Saved Successfully")