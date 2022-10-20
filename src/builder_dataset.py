import numpy as np
import pywt
from multiprocessing import Pool
import re
import scipy.io as sio
import os
from os.path import join
import scipy.stats
from sklearn.model_selection import train_test_split
import pickle
import scipy
import scipy.fftpack

__author__ = "Diego Cabrera"
__copyright__ = "Copyright 2018, The GIDTEC Fault Diagnosis Project"
__credits__ = ["Diego Cabrera"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Diego Cabrera"
__email__ = "dcabrera@ups.edu.ec"
__status__ = "Prototype"


def signal2wp_energy(signal, wavelets, max_level):
    signal = signal.squeeze()
    energy_coef = np.zeros((len(wavelets),2**max_level))
    for j, wavelet in enumerate(wavelets):
        wp_tree = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=max_level)
        level = wp_tree.get_level(max_level, order='freq')

        for i, node in enumerate(level):
            energy_coef[j,i] = np.sqrt(np.sum(node.data ** 2)) / node.data.shape[0]

    return energy_coef.flatten()

def signal2wp_spectrum(signal, wavelet, max_level):
    signal = signal.squeeze()
    wp_tree = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=max_level)
    nodes = wp_tree.get_level(max_level, order='freq')
    spectrum = np.array([np.abs(n.data) for n in nodes])
    #spectrum = np.log(spectrum+0.1)
    #mean = spectrum.mean()
    #spectrum = spectrum - mean
    #max = np.abs(spectrum).max()
    #spectrum = spectrum / max

    return spectrum


def rectified_average(signal):
    """
    Compute rectified average feature
    :param signal: time-series
    :type signal: numpy array
    :return: rectified average of signal
    :rtype: float
    """
    return np.mean(np.abs(signal))


def statistic_features(signal):
    """
    Compute a group of statistical features
    :param signal: time-series
    :type signal: numpy array
    :return: group of statistical feature from the signal
    :rtype: tuple
    """
    rms = np.sqrt(np.mean(np.square(signal)))
    std_dev = np.std(signal)
    kurtosis = scipy.stats.kurtosis(signal)
    peak = np.max(signal)
    crest = peak / rms
    r_mean = rectified_average(signal)
    form = rms / r_mean
    impulse = peak / r_mean
    variance = std_dev ** 2
    skewness = scipy.stats.skew(signal)
    square_root_amplitude = np.mean(np.sqrt(np.abs(signal))) ** 2
    clearance = peak / square_root_amplitude
    return crest, form, r_mean, square_root_amplitude, kurtosis, variance, clearance, impulse, skewness

def cut_signal(signal, length, step):
    i = 0
    signals = []
    while i + length <= signal.shape[0]:
        signals.append(signal[i:i + length])
        i += step
    signals = np.array(signals)
    return signals

def matrix_builder(path_in,path_out):
    data = sio.loadmat(path_in)
    for sensor in [0,1,2,3,4,5,6,7,8,9,10,11]:
        features_train = []
        labels_train = []
        features_test = []
        labels_test = []
        for severity in range(1,9):
            print('severity',severity,'sensor',sensor)
            for repetition in range(3):
                signal = data['A'+str(severity)][:,sensor,repetition]
                signals = cut_signal(signal,1620,100)
                if repetition < 2:
                    features_train.extend(
                        [signal2wp_energy(
                            chunk, ['db7', 'sym3'], 6) for chunk in signals])
                    labels_train.extend(signals.shape[0]*[severity -1])
                else:
                    features_test.extend(
                        [signal2wp_energy(
                            chunk, ['db7', 'sym3'], 6) for chunk in signals])
                    labels_test.extend(signals.shape[0]*[severity - 1])
        features_train = np.array(features_train)
        labels_train = np.array(labels_train)
        features_test = np.array(features_test)
        labels_test = np.array(labels_test)

        with open(path_out+'/dataset_A_sensor'+str(sensor)+'.pkl','wb') as f:
            pickle.dump((features_train,labels_train,features_test,labels_test),
                        f,protocol=pickle.HIGHEST_PROTOCOL)

def spectrum_dataset(path_in,path_out):
    data = sio.loadmat(path_in)
    severity = 8
    for num_fault,fault in enumerate(['A','B','C','D','E','F','G','H','I','J','K','L','P']):
        if fault == 'P':
            data = sio.loadmat('/home/titan/Dropbox/Postdoc_China/Data_3D_Printer/R75/DataP.mat')
            severity = 0
        for sensor in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            spectrum_train = []
            labels_train = []
            spectrum_test = []
            labels_test = []
            print('fault', fault, 'sensor', sensor)

            for repetition in range(3):
                signal = data[fault + str(severity)][:, sensor, repetition]
                signals = cut_signal(signal, 1620, 10)
                if repetition < 2:
                    spectrum_train.extend(
                        [signal2wp_spectrum(
                            chunk, 'db7', 6) for chunk in signals])
                    labels_train.extend(signals.shape[0] * [num_fault])
                else:
                    spectrum_test.extend(
                        [signal2wp_spectrum(
                            chunk, 'db7', 6) for chunk in signals])
                    labels_test.extend(signals.shape[0] * [num_fault])
            spectrum_train = np.array(spectrum_train)
            labels_train = np.array(labels_train)
            spectrum_test = np.array(spectrum_test)
            labels_test = np.array(labels_test)

            with open(path_out + '/dataset_'+fault+'_sensor' + str(sensor) + '.pkl', 'wb') as f:
                pickle.dump((spectrum_train, labels_train, spectrum_test, labels_test),
                            f, protocol=pickle.HIGHEST_PROTOCOL)

def time_dataset(path_in,path_out):
    data = sio.loadmat(path_in)
    severity = 3 #8
    for num_fault,fault in enumerate(['M','N','O']):#enumerate(['A','B','C','D','E','F','G','H','I','J','K','L','P']):
        if fault == 'P':
            data = sio.loadmat('/home/titan/Dropbox/Postdoc_China/Data_3D_Printer/R75/DataP.mat')
            severity = 0
        for sensor in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            time_train = []
            labels_train = []
            time_test = []
            labels_test = []
            print('fault', fault, 'sensor', sensor)

            for repetition in range(3):
                signal = data[fault + str(severity)][:, sensor, repetition]
                signals = cut_signal(signal, 1620, 10)
                if repetition < 2:
                    time_train.extend(signals.tolist())
                    labels_train.extend(signals.shape[0] * [num_fault])
                else:
                    time_test.extend(signals.tolist())
                    labels_test.extend(signals.shape[0] * [num_fault])
            time_train = np.array(time_train)
            labels_train = np.array(labels_train)
            time_test = np.array(time_test)
            labels_test = np.array(labels_test)

            with open(path_out + '/dataset_'+fault+'_sensor' + str(sensor) + '.pkl', 'wb') as f:
                pickle.dump((time_train.reshape((-1,1,1620)), labels_train, time_test.reshape((-1,1,1620)), labels_test),
                            f, protocol=pickle.HIGHEST_PROTOCOL)
def compute_features(signal):
    features_time = statistic_features(signal)
    yf = scipy.fftpack.fft(signal)
    yf = 2.0 / len(signal) * np.abs(yf[:len(signal) // 2])
    features_freq = statistic_features(yf)
    wp_tree = pywt.WaveletPacket(data=signal, wavelet='db7', maxlevel=5)
    nodes = wp_tree.get_level(5, order='freq')
    features_tf = np.array([statistic_features(n.data) for n in nodes]).flatten()
    features = np.concatenate((features_time,features_freq,features_tf))
    return features


def features_dataset(path_in,path_out):
    data = sio.loadmat(path_in)
    severity = 3
    for num_fault,fault in enumerate(['P']):#enumerate(['A','B','C','D','E','F','G','H','I','J','K','L','P']):
        if fault == 'P':
            data = sio.loadmat('/home/titan/Dropbox/Postdoc_China/Data_3D_Printer/R75/DataP.mat')
            severity = 0
        features_sensors_train = []
        features_sensors_test = []
        for sensor in [3,4,5,6,7,8]:
            features_train = []
            labels_train = []
            features_test = []
            labels_test = []
            print('fault', fault, 'sensor', sensor)

            for repetition in range(3):
                signal = data[fault + str(severity)][:, sensor, repetition]
                signals = cut_signal(signal, 1620, 10)
                if repetition < 2:
                    features_train.extend([compute_features(chunk) for chunk in signals])
                    labels_train.extend(signals.shape[0] * [num_fault])
                else:
                    features_test.extend([compute_features(chunk) for chunk in signals])
                    labels_test.extend(signals.shape[0] * [num_fault])
            features_train = np.array(features_train)
            labels_train = np.array(labels_train)
            features_test = np.array(features_test)
            labels_test = np.array(labels_test)
            features_sensors_train.append(features_train)
            features_sensors_test.append(features_test)
        with open(path_out + '/data_encoded_f_'+fault+'.pkl', 'wb') as f:
            pickle.dump((np.hstack(np.array(features_sensors_train)), np.hstack(np.array(features_sensors_test))),
                        f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    path_in = '/home/titan/Dropbox/Postdoc_China/Data_3D_Printer/R75/DataM_O.mat'
    # For spectrum
    # path_out = 'Printer3D_spectrum'

    # For features
    path_out = '../results'

    # For time-series
    #path_out = 'Printer3D_time'

    if not os.path.exists('../data/' + path_out):
        os.makedirs('../data/' + path_out)
    # Datasets generation process
    features_dataset(path_in, '../data/' + path_out)