import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open('../results/nu1.pkl','rb') as f:
    results_gamma = pickle.load(f)
print(results_gamma)