import soundfile  
import numpy as np
import librosa  
import glob
import os
import pickle  # model mentése a betanitás után
from sklearn.model_selection import train_test_split  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score  # milyen a model


def extract_feature(file_name, **kwargs):
  
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result


# minden érzelem a a beatanitási hangokbol
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# am,iket én nézek
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}


def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("data/Actor_*/*.wav"):
        # melyik fájl
        basename = os.path.basename(file)
        emotion = int2emotion[basename.split("-")[2]]
        # csak a beállított AVAILABLE_EMOTIONS-t engedélyezzük
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        X.append(features)
        y.append(emotion)
    
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)


# 75% tréning 25% teszt
X_train, X_test, y_train, y_test = load_data(test_size=0.25)



print("A betanításhoz felhasznált minták száma:", X_train.shape[0])

print("A teszteléshez felhasznált minták száma:", X_test.shape[0])

print("Number of features:", X_train.shape[1])

# legjobb modell Grid Search alapján
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}

# Multi Layer Perceptron classifier

model = MLPClassifier(**model_params)

# betanitás
print("A modell tanul ...")
model.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# mennyire pontos a modell
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Pontosság: {:.2f}%".format(accuracy * 100))

# modell elmentése
# eredménykönyvtár létrehozása, ha még nem létezik
if not os.path.isdir("result"):
    os.mkdir("result")

# ONNX formátumba konvertálás
from skl2onnx import to_onnx

onx = to_onnx(model, X_test)
with open("result/mlp_classifier.onnx", "wb") as f:
    f.write(onx.SerializeToString())

pickle.dump(model, open("result/mlp_classifier.model", "wb"))
