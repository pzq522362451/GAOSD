import tensorflow as tf
import pickle
from data_provider import read_data_sets
import pywt
import numpy as np

def signal2wp_energy(signal, wavelets, max_level):
    signal = signal.squeeze()
    energy_coef = np.zeros((len(wavelets),2**max_level))
    for j, wavelet in enumerate(wavelets):
        wp_tree = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=max_level)
        level = wp_tree.get_level(max_level, order='freq')
        for i, node in enumerate(level):
            energy_coef[j,i] = np.sqrt(np.sum(node.data ** 2)) / node.data.shape[0]

    return energy_coef.flatten()

root_data = '../dataset/sensors/'

faults = ['C0','C2','C3','C4','C5','C6','C7','C8']
# faults = ['C0']
sensors = [9,10,11]
for fault in faults:
    b = np.zeros((4752, 1250))
    b2 = np.zeros((2376, 1250))
    path_out = '../results/wpt/data_encoded_f_'+fault+'_91011.pkl'
    train_encoded = []
    test_encoded = []
    for sensor in sensors:

        data_dir = root_data+'dataset_'+fault+'_R01_to_R03_' + str(sensor) + '.pkl'
        print(data_dir)
        printer_data,printer_test = read_data_sets(data_dir)
        b = np.concatenate((b, printer_data.images), axis=1)
        b2 = np.concatenate((b2, printer_test.images), axis=1)
    b = b[:, 1250:]
    b2 = b2[:, 1250:]

    # # 小波包变换，db7，分解7层
    wpt1 = []
    for k in range(b.shape[0]):
        wpt1.append(signal2wp_energy(b[k, :], wavelets=['db7'], max_level=7))
    wpt_train = np.array(wpt1).reshape(b.shape[0], 128)

    wpt2 = []
    for k in range(b2.shape[0]):
        wpt2.append(signal2wp_energy(b2[k, :], wavelets=['db7'], max_level=7))
    wpt_test = np.array(wpt2).reshape(b2.shape[0], 128)

    with open(path_out,'wb') as f:
        pickle.dump((wpt_train,wpt_test),f,pickle.HIGHEST_PROTOCOL)