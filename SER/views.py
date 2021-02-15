from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import numpy as np
import librosa
from playsound import playsound
import sounddevice
from scipy.io.wavfile import write
import os

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050
RECORD_TIME = 1


class _Speech_Recognition_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = None
    _mapping = [
        "OAF_angry",
        "OAF_disgust",
        "OAF_Fear",
        "OAF_happy",
        "OAF_neutral",
        "OAF_Pleasant_surprise",
        "OAF_Sad",
        "YAF_angry",
        "YAF_disgust",
        "YAF_fear",
        "YAF_happy",
        "YAF_neutral",
        "YAF_pleasant_surprised",
        "YAF_sad"
    ]
    _instance = None

    def predict(self, file_path):
        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted = np.argmax(predictions)
        predicted_result = self._mapping[predicted]
        return predicted_result

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        signal, sample_rate = librosa.load(file_path)
        #aa, bb = librosa.effects.trim(signal, top_db=30)
        #print(aa, bb)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Speech_Recognition_Service():
    # ensure an instance is created only the first time the factory function is called
    if _Speech_Recognition_Service._instance is None:
        _Speech_Recognition_Service._instance = _Speech_Recognition_Service()
        _Speech_Recognition_Service.model = tf.keras.models.load_model(
            SAVED_MODEL_PATH)
    return _Speech_Recognition_Service._instance


def record_sound():
    print("Recording...Say something")
    recorded_sound = sounddevice.rec(int(
        RECORD_TIME * SAMPLES_TO_CONSIDER), samplerate=SAMPLES_TO_CONSIDER, channels=2)
    sounddevice.wait()
    write("sample3.wav", SAMPLES_TO_CONSIDER, recorded_sound)


# Create your views here.


def home(request):
    return render(request, "SER/index.html")


def record(request):

    if request.method == 'POST':
        audio_clip = request.POST['audio_clip']

        playsound(audio_clip)
        print("post request running")

        # record_sound()

        # playsound('sample3.wav')

        # create 2 instances of the keyword spotting service
        model = Speech_Recognition_Service()
        model1 = Speech_Recognition_Service()

        # check that different instances of the keyword spotting service point back to the same object (singleton)
        assert model is model1

        # make a prediction
        result = model.predict(audio_clip)
        # os.remove("sample3.wav")
        print(result)

        return render(request, "SER/try.html", {"result": result})
    else:
        return render(request, "SER/record.html")
