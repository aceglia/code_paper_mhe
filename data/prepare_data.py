from pyomeca import Analogs, Markers
import scipy.io as sio
import numpy as np
import bioviz
from biosiglive.server import Server
import os
import biorbd
from biosiglive.data_processing import read_data, add_data_to_pickle
import matplotlib.pyplot as plt

subject = "Remi"
scaled = True
scal = "_scaled" if scaled else ""
trial = "test_abd"
mat = read_data(f"/home/amedeo/Documents/programmation/data_article/{subject}/{trial}")
emg = mat["emg_proc"]
# markers = mat["kin_target"]
markers = mat["markers"]
# emg = mat["muscles_target"]

plt.figure("Muscles")
# for i in muscle_track_idx:
# for i in range(emg.shape[0]):
#     plt.subplot(4, 5, i + 1)
#     plt.plot(emg[i, :])
# plt.show()
model_path = f"/home/amedeo/Documents/programmation/data_article/{subject}/Wu_Shoulder_Model_mod_wt_wrapp_{subject}{scal}.bioMod"

model = biorbd.Model(model_path)
# q_recons = mat["kalman"]
q_recons, _ = Server.kalman_func(markers, model=model)
# print(q_recons[5, :])
# q_recons[5, :] = q_recons[5, :] - np.pi
# q_recons[7, :] = q_recons[7, :] + np.pi
# q_recons = mat["kalman"]
b = bioviz.Viz(model_path=model_path, show_floor=False, show_muscles=True, show_gravity_vector=False)
b.load_experimental_markers(markers)
b.load_movement(q_recons)
b.exec()

data_to_save = {"emg": emg, "markers": markers, "kalman": q_recons[:, :]}
# data_to_save = {"emg": emg, "markers": markers, "kalman": mat["kalman"]}
data_path = f"/home/amedeo/Documents/programmation/data_article/{subject}/{trial}{scal}"
if os.path.isfile(data_path):
    os.remove(data_path)
add_data_to_pickle(data_to_save, f"/home/amedeo/Documents/programmation/data_article/{subject}/{trial}{scal}")

subject = "Remi"
# data_dir = f"test_09_12_21/{subject}/"
data_dir = f"/home/amedeo/Documents/programmation/data_article/{subject}/"

# # DA, DM, DP, TS, TM, TI, LAT, PEC, BIC, TRimed
# muscle_names = ["pec.IM EMG1", "DA.IM EMG2", "DM.IM EMG3", "DP.IM EMG4", "bic.IM EMG5", "tri.IM EMG6",
#                 "TRAPsup.IM EMG7", "TRAPmed.IM EMG8", "TRAPinf.IM EMG9", "lat.IM EMG10"]
muscle_names = ["pec.IM EMG1", "DA.IM EMG2", "DM.IM EMG3", "DP.IM EMG4", "bic.IM EMG5", "tri.IM EMG6",
                "trap_sup.IM EMG7", "trap_med.IM EMG8", "trap_inf.IM EMG9", "lat.IM EMG10"]

# # --- MVC --- #
# mvc_list = ["MVC_bic", "MVC_DA", "MVC_DM", "MVC_DP", "MVC_lat", "MVC_pec", "MVC_TI", "MVC_TM", "MVC_tri",
#             "MVC_trap_sup"]
#
# mvc_list_max = np.ndarray((len(muscle_names), 2000))
# mvc_list_val = np.ndarray((len(muscle_names), 2))
# for i in range(len(mvc_list)):
#     b = Analogs.from_c3d(data_dir + f"{str(mvc_list[i])}.c3d", usecols=muscle_names)
#     mvc_temp = (
#     b.meca.band_pass(order=4, cutoff=[10, 425])
#     .meca.center()
#     .meca.abs()
#     .meca.low_pass(order=4, cutoff=5, freq=b.rate)
#     # .meca.normalize()
#     )
#     for j in range(len(muscle_names)):
#         mvc_list_val = 1
#     mvc_temp = -np.sort(-mvc_temp.data, axis=1)
#     if i == 0:
#         mvc_list_max = mvc_temp[:, :2000]
#     else:
#         mvc_list_max = np.concatenate((mvc_list_max, mvc_temp[:, :2000]), 1)
#
# mvc_list_max = -np.sort(-mvc_list_max, 1)[:, :2000]
# mvc_list_max = np.mean(mvc_list_max, 1)

trial_name = "fonction"
data_path = data_dir + trial_name
a = Analogs.from_c3d(f"{data_path}.c3d", usecols=muscle_names)
emg = a
# emg_rate = int(a.rate)
# emg = (
#     a.meca.band_pass(order=4, cutoff=[10, 425])
#     .meca.center()
#     .meca.abs()
#     .meca.low_pass(order=4, cutoff=5, freq=a.rate)
#     # .meca.normalize()
# )
# emg = emg[:, ::20]
# sio.savemat(f"./sujet_{sujet}/mvc_sujet_{sujet}.mat", {'mvc_treat': mvc_list_max})
# sio.savemat(data_dir + "MVC.mat", {'mvc_treat': mvc_list_max})
model = data_dir + f'Wu_Shoulder_Model_mod_wt_wrapp_{subject}.bioMod'
# model = data_dir + f'Wu_Shoulder_Model_mod_wt_wrapp_{subject}.bioMod'
# model = os.path.dirname(os.getcwd()) + '/models/arm_wt_rot_scap.bioMod'
# model = "/home/amedeo/Documents/programmation/RT_Optim/models/wu_model.bioMod"
bmodel = biorbd.Model(model)

# emg_norm_tmp = np.ndarray((len(muscle_names), emg.shape[1]))
# emg_norm = np.zeros((bmodel.nbMuscles(), emg.shape[1]))
# for i in range(len(muscle_names)):
#     emg_norm_tmp[i, :] = emg[i, :]/mvc_list_max[i]

# for i in range(len(bmodel.nbMuscles())):
#     print(bmodel.muscleNames()[i].to_string())
# emg_norm[13, :] = emg_norm_tmp[0, :]
# emg_norm[15, :] = emg_norm_tmp[1, :]
# emg_norm[21, :] = emg_norm_tmp[2, :]
# emg_norm[10, :] = emg_norm_tmp[3, :]
# emg_norm[1, :] = emg_norm_tmp[4, :]
# emg_norm[2, :] = emg_norm_tmp[5, :]
# emg_norm[27, :] = emg_norm_tmp[6, :]
# emg_norm[[14, 25, 26], :] = emg_norm_tmp[7, :]
# emg_norm[[23,24], :] = emg_norm_tmp[8, :]
# emg_norm[[28,29,30], :] = emg_norm_tmp[9, :]


# --- Markers --- #
marker_names = []
for i in range(len(bmodel.markerNames())):
    marker_names.append(bmodel.markerNames()[i].to_string())

markers_full = Markers.from_c3d(f"{data_path}.c3d", usecols=marker_names)
markers_full = markers_full[:, :, :].data
q_recons, _ = Server.kalman_func(markers_full * 0.001, model=bmodel)
markers_full = np.nan_to_num(markers_full)
# markers = np.ndarray((3, markers_full.shape[1], markers_full.shape[2]))
# markers_exp = np.ndarray((3, markers_full.shape[1], markers_full.shape[2]))
# markers_forw = np.ndarray((3, markers_full.shape[1], markers_full.shape[2]))
#
# for f in range(markers_full.shape[2]):
#     markers = np.array([mark.to_array() for mark in bmodel.markers(q_recons[:, f])]).T
#     for i in range(markers_full.shape[1]):
#         # if f !=0:
#         if np.product(markers_full[:, i, f]) == 0:
#             markers_exp[:3, i, f] = markers_exp[:3, i, f-1]
#             markers_forw[:3, i, f] = markers[:3, i]
#         else:
#             markers_exp[:3, i, f] = markers_full[:3, i, f]
#             markers_forw[:3, i, f] = markers_full[:3, i, f]
# #
b = bioviz.Viz(model_path=model, show_floor=False, show_muscles=True, show_gravity_vector=False)
b.load_experimental_markers(markers_full * 0.001)
b.load_movement(q_recons)
b.exec()

sio.savemat(data_dir + f"test_{trial_name}_for_server.mat", {'emg': emg, "markers": markers_full, "kalman": q_recons})
