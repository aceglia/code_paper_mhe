from pyomeca import Analogs, Markers
import scipy.io as sio
import numpy as np
import bioviz
from biosiglive.server import Server
from scipy import interpolate
import os
import biorbd
from biosiglive.data_processing import read_data, add_data_to_pickle
import matplotlib.pyplot as plt

# base_dir = "/home/amedeo/Documents/programmation/code_paper_mhe/results/"
model_dir = "/home/amedeo/Documents/programmation/code_paper_mhe/data/test_09_02_22/"
base_dir = model_dir
subject = "Etienne"

scaled = True
scal = "_scaled" if scaled else ""

trial = "test_anato_abd"

mat = read_data(f"{base_dir}{subject}/{trial}")
muscles_target = mat["emg_proc"]
markers_target = mat["markers"]
# # x_ref
# t_f = markers.shape[2] / 8
# interp_size = int(100/8)
# markers_target = np.zeros(
#     (3, markers.shape[1], int(markers.shape[2] * interp_size)))
# for i in range(3):
#     x = np.linspace(0, t_f, markers.shape[2])
#     f_mark = interpolate.interp1d(x, markers[i, :, :])
#     x_new = np.linspace(0, t_f, int(markers_target.shape[2]))
#     markers_target[i, :, :] = f_mark(x_new)
#
# # muscle_target
# x = np.linspace(0, t_f, emg.shape[1])
# f_mus = interpolate.interp1d(x, emg)
# x_new = np.linspace(0, t_f, int(emg.shape[1] * interp_size))
# muscles_target = f_mus(x_new)
#
# plt.figure("markers")
# for i in range(markers.shape[1]):
#     plt.subplot(4, 4, i+1)
#     plt.plot(x, markers[0, i, :].T)
#     plt.plot(x, markers[1, i, :].T)
#     plt.plot(x, markers[2, i, :].T)
#     plt.plot(x_new, markers_target[0, i, :].T)
#     plt.plot(x_new, markers_target[1, i, :].T)
#     plt.plot(x_new, markers_target[2, i, :].T)
#
# plt.figure("Muscles")
# # for i in muscle_track_idx:
# for i in range(muscles_target.shape[0]):
#     plt.subplot(5, 4, i + 1)
#     plt.plot(x_new, muscles_target[i, :])
#     plt.plot(x, emg[i, :], '-r')
#
# plt.show()
# q_recons = mat["kalman"]
trial = 'anato_abd_prepared'
model_path = f"{model_dir}{subject}/Wu_Shoulder_Model_mod_wt_wrapp_{subject}{scal}.bioMod"

model = biorbd.Model(model_path)

# kalman
q_recons, _ = Server.kalman_func(markers_target, model=model)
# q_recons[5, :] = q_recons[5, :] - 2*np.pi
# q_recons[7, :] = q_recons[7, :] + 2*np.pi
# Viz
# b = bioviz.Viz(model_path=model_path, show_floor=False, show_muscles=True, show_gravity_vector=False)
# b.load_experimental_markers(markers_target)
# b.load_movement(q_recons)
# b.exec()

# save
data_to_save = {"emg": muscles_target, "markers": markers_target, "kalman": q_recons[:, :]}
data_path = f"{base_dir}{subject}/{trial}{scal}"
if os.path.isfile(data_path):
    os.remove(data_path)
add_data_to_pickle(data_to_save, f"{base_dir}{subject}/{trial}{scal}")