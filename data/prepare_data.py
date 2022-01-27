from pyomeca import Analogs, Markers
import scipy.io as sio
import numpy as np
import bioviz
from biosiglive.server import Server
import os
import biorbd
from biosiglive.data_processing import read_data, add_data_to_pickle
import matplotlib.pyplot as plt

base_dir = "/home/amedeo/Documents/programmation/data_article/"

subject = "Remi"

scaled = True
scal = "_scaled" if scaled else ""

trial = "test_abd"

mat = read_data(f"{base_dir}{subject}/{trial}")
emg = mat["muscles_target"]
markers = mat["kin_target"]
# q_recons = mat["kalman"]

model_path = f"{base_dir}{subject}/Wu_Shoulder_Model_mod_wt_wrapp_{subject}{scal}.bioMod"

model = biorbd.Model(model_path)

# kalman
q_recons, _ = Server.kalman_func(markers, model=model)

# Viz
b = bioviz.Viz(model_path=model_path, show_floor=False, show_muscles=True, show_gravity_vector=False)
b.load_experimental_markers(markers)
b.load_movement(q_recons)
b.exec()

# save
data_to_save = {"emg": emg, "markers": markers, "kalman": q_recons[:, :]}
data_path = f"{base_dir}{subject}/{trial}{scal}"
if os.path.isfile(data_path):
    os.remove(data_path)
add_data_to_pickle(data_to_save, f"{base_dir}{subject}/{trial}{scal}")