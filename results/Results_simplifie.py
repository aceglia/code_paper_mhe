import numpy as np
from biosiglive.data_processing import read_data
from biosiglive.server import Server
try :
    import biorbd
except :
    import biorbd_casadi as biorbd
import os
from pyomeca import Markers
import matplotlib.pyplot as plt
use_torque = True
animate = False
import glob

parent = os.path.dirname(os.getcwd())
subject = ["Jules"]#, "Mathis"]#, "Clara"]
trial = ["result_abd"]#, "test_abd_full"]
result_mat = []
models = []
for subj in subject:
    result_dir = f"/home/amedeo/Documents/programmation/data_article/{subj}/"
    for trials in trial:
        file = result_dir + f"{trials}"
        result_mat.append(read_data(file))
    models.append(biorbd.Model(result_dir + f"Wu_Shoulder_Model_mod_wt_wrapp_{subj}_scaled.bioMod"))

muscle_track_idx = [14, 25, 26,  # PEC
                    13,  # DA
                    15,  # DM
                    21,  # DP
                    23, 24,  # bic
                    28, 29, 30,  # tri
                    10,  # TRAPsup
                    2,  # TRAPmed
                    3,  # TRAPinf
                    27  # Lat
                    ]

plt.figure("Q")
for s in range(len(subject)):
    for i in range(models[0].nbQ()):
        plt.subplot(3, 3, i+1)
        plt.plot(result_mat[s]["X_est"][i, :] * 180/np.pi)
        plt.plot(result_mat[s]["kalman"][i, :] * 180/np.pi, '-r')
        if s == 0:
            plt.title(models[0].nameDof()[i].to_string())


plt.figure("torque")
for s in range(len(subject)):
    for i in range(models[0].nbQ()):
        plt.subplot(3, 3, i + 1)
        plt.plot(result_mat[s]["tau_est"][i, :])

plt.figure("Forces")
for s in range(len(subject)):
    for i in range(models[0].nbMuscles()):
        plt.subplot(5, 7, i + 1)
        plt.plot(result_mat[s]["f_est"][i, :])
        if s == 0:
            plt.title(models[0].muscleNames()[i].to_string())

for s in range(len(subject)):
    plt.figure("markers")
    markers = np.ndarray((3, result_mat[s]["kin_target"].shape[1], result_mat[s]["kin_target"].shape[2]))
    for i in range(result_mat[s]["kin_target"].shape[2]):
        markers[:, :, i] = np.array([mark.to_array() for mark in models[s].markers(result_mat[s]["X_est"][:, i])]).T
    for i in range(result_mat[s]["kin_target"].shape[1]):
        plt.subplot(4, 4, i+1)
        plt.plot(result_mat[s]["kin_target"][0, i, :].T * 100, "-r")
        plt.plot(result_mat[s]["kin_target"][1, i, :].T * 100, "-r")
        plt.plot(result_mat[s]["kin_target"][2, i, :].T * 100, "-r")
        plt.plot(markers[0, i, :].T * 100, "--b")
        plt.plot(markers[1, i, :].T * 100, "--b")
        plt.plot(markers[2, i, :].T * 100, "--b")

plt.figure("Muscles")
for s in range(len(subject)):
    # for i in muscle_track_idx:
    for i in range(models[0].nbMuscles()):
        plt.subplot(7, 5, i + 1)
        if i in muscle_track_idx:
            idx = muscle_track_idx.index(i)
            plt.plot(result_mat[s]["muscles_target"][idx, :], '-r')
        plt.plot(result_mat[s]["U_est"][i, :])
        if s == 0:
            plt.title(models[0].muscleNames()[i].to_string())
        plt.tight_layout()

plt.show()

