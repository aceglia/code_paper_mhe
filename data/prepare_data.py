from pyomeca import Analogs, Markers
import scipy.io as sio
import numpy as np
import bioviz
from biosiglive.server import Server
import os
import biorbd

data_dir = "data_30_11_21/RT_test/"

# # DA, DM, DP, TS, TM, TI, LAT, PEC, BIC, TRimed
muscle_names = ["DA.IM EMG2", "DM.IM EMG3", "DP.IM EMG4", "trap_sup.IM EMG7", "trap_med.IM EMG8", "trap_inf.IM EMG9",
                "lat.IM EMG10", "pec.IM EMG1", "bic.IM EMG5", "tri.IM EMG6"]


# muscle_names = ["Sensor 1.IM EMG1", "Sensor 10.IM EMG10", "Sensor 2.IM EMG2", "Sensor 3.IM EMG3", "Sensor 4.IM EMG4",
#                 "Sensor 5.IM EMG5", "Sensor 6.IM EMG6", "Sensor 7.IM EMG7", "Sensor 8.IM EMG8", "Sensor 9.IM EMG9"]

# # --- MVC --- #
mvc_list = ["MVC_bic", "MVC_DA", "MVC_DM", "MVC_DP", "MVC_lat", "MVC_pec", "MVC_TI", "MVC_TM", "MVC_tri",
            "MVC_trap_sup"]

mvc_list_max = np.ndarray((len(muscle_names), 2000))
mvc_list_val = np.ndarray((len(muscle_names), 2))
for i in range(len(mvc_list)):
    b = Analogs.from_c3d(data_dir + f"{str(mvc_list[i])}.c3d", usecols=muscle_names)
    mvc_temp = (
    b.meca.band_pass(order=4, cutoff=[10, 425])
    .meca.center()
    .meca.abs()
    .meca.low_pass(order=4, cutoff=5, freq=b.rate)
    # .meca.normalize()
    )
    for j in range(len(muscle_names)):
        mvc_list_val = 1
    mvc_temp = -np.sort(-mvc_temp.data, axis=1)
    if i == 0:
        mvc_list_max = mvc_temp[:, :2000]
    else:
        mvc_list_max = np.concatenate((mvc_list_max, mvc_temp[:, :2000]), 1)

mvc_list_max = -np.sort(-mvc_list_max, 1)[:, :2000]
mvc_list_max = np.mean(mvc_list_max, 1)
muscle_names = ["pec.IM EMG1", "DA.IM EMG2", "DM.IM EMG3", "DP.IM EMG4", "bic.IM EMG5", "tri.IM EMG6",
                "trap_sup.IM EMG7", "trap_med.IM EMG8", "trap_inf.IM EMG9", "lat.IM EMG10", ]

trial_name = "flex"
data_path = data_dir + trial_name
a = Analogs.from_c3d(f"{data_path}.c3d", usecols=muscle_names)
emg_rate = int(a.rate)
emg = (
    a.meca.band_pass(order=4, cutoff=[10, 425])
    .meca.center()
    .meca.abs()
    .meca.low_pass(order=4, cutoff=5, freq=a.rate)
    # .meca.normalize()
)
emg = emg[:, ::20]
# sio.savemat(f"./sujet_{sujet}/mvc_sujet_{sujet}.mat", {'mvc_treat': mvc_list_max})
# sio.savemat(data_dir + "MVC.mat", {'mvc_treat': mvc_list_max})
model = os.path.dirname(os.getcwd()) + '/data/data_30_11_21/Wu_Shoulder_Model_mod_wt_wrapp_Jules.bioMod'
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
# markers_full_names = ["STER", "XIPH", "T10", "C7", "CLAV_SC", "CLAV_AC", "SCAP_IA", "Acrom", "SCAP_AA", "EPICl",
#                       "EPICm", "DELT", "ARMl",  "STYLu", "LARM_elb", "STYLr"]
marker_names = []
for i in range(len(bmodel.markerNames())):
    marker_names.append(bmodel.markerNames()[i].to_string())
# "Amedeo:SCAP_CP", , "Amedeo:T1"
# markers_full_names = ["ASISr","PSISr", "PSISl","ASISl","XIPH","STER","STERlat","STERback","XIPHback","ThL",
#            "CLAV_SC","CLAV_AC","SCAP_CP","SCAP_AA","SCAPspine","SCAP_TS","SCAP_IA","DELT","ARMl",
#            "EPICl","EPICm","LARM_ant","LARM_post","LARM_elb","LARM_dist","STYLrad","STYLrad_up","STYLulna_up",
#            "STYLulna","META2dist","META2prox","META5prox","META5dist","MAIN_opp"]
# markers_full_names = ["CLAV_SC","CLAV_AC","SCAP_CP","SCAP_AA","SCAP_TS","SCAP_IA","DELT","ARMl","EPICl",
#            "EPICm","LARM_ant","LARM_post","LARM_elb","LARM_dist"]

markers_full = Markers.from_c3d(f"{data_path}.c3d", usecols=marker_names)
markers_full = markers_full[:, :, :].data * 0.001
markers_full = np.nan_to_num(markers_full)

q_recons, _ = Server.kalman_func(markers_full, model=bmodel)
markers = np.ndarray((3, markers_full.shape[1], markers_full.shape[2]))
markers_exp = np.ndarray((3, markers_full.shape[1], markers_full.shape[2]))
markers_forw = np.ndarray((3, markers_full.shape[1], markers_full.shape[2]))

for f in range(markers_full.shape[2]):
    markers = np.array([mark.to_array() for mark in bmodel.markers(q_recons[:, f])]).T
    for i in range(markers_full.shape[1]):
        # if f !=0:
        if np.product(markers_full[:, i, f]) == 0:
            markers_exp[:3, i, f] = markers_exp[:3, i, f-1]
            markers_forw[:3, i, f] = markers[:3, i]
        else:
            markers_exp[:3, i, f] = markers_full[:3, i, f]
            markers_forw[:3, i, f] = markers_full[:3, i, f]
#
b = bioviz.Viz(model_path=model, show_floor=False, show_muscles=True, show_gravity_vector=False)
b.load_movement(q_recons)
b.load_experimental_markers(markers_forw)
b.exec()

sio.savemat(data_dir + f"test_{trial_name}.mat", {'emg': emg, "markers": markers_forw, "kalman": q_recons})
