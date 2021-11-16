import numpy as np
from biosiglive.data_processing import read_data
from biosiglive.server import Server
import biorbd
import os
from pyomeca import Markers

use_torque = True
animate = False

parent = os.path.dirname(os.getcwd())
file_name = parent + "/" + "results/results_20211116/Results_MHE_q_EMG_act_torque_driven_test_20211116-1331"
model = biorbd.Model(parent + "/models/wu_model.bioMod")
c3d = parent + "data/data_09_21/abd.c3d"
mat = read_data(file_name)

nbGT = model.nbGeneralizedTorque() if use_torque is True else 0

# Recons markers with kalman:
# q_recons, q_dot = Server.kalman_func(mat["kin_target"][:, :, :], model=model)

# Markers from c3d
# markers = Markers.from_c3d(c3d).values
# markers = markers*0.001

if animate is True:
    import bioviz
    b = bioviz.Viz(loaded_model=model)
    # b.load_experimental_markers(mat["kin_target"])
    # b.load_experimental_markers(c3d)
    b.load_movement(mat["X_est"][:model.nbQ(), :])
    b.exec()

print(np.mean(mat["sol_freq"][:500]))
print(np.std(mat["sol_freq"][:500]))
# print(mat)
import matplotlib.pyplot as plt

plt.figure("Q")
for i in range(0,int(mat["X_est"].shape[0]/2)):
    plt.plot(mat["X_est"][i, :]*180/np.pi)
    # if len(mat["kin_target"].shape) == 2:
    #     plt.plot(mat["kin_target"][i, :]*180/np.pi, '0')
    plt.plot(mat["kalman"][i, :]*180/np.pi, 'x')
    # plt.plot(q_recons[i, :] * 180 / np.pi, 'x')

# plt.figure("markers")
# for i in range(3):
#     plt.plot(mat["markers_target"][i, :, :].T)
muscle_track_idx = [0, 1, 2, 3, 4, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26]
col = 4
lin = int(len(muscle_track_idx)/col)
count = 0
plt.figure("Muscles")
for i in muscle_track_idx:
    plt.subplot(lin, col, count + 1)
    if isinstance(mat["muscles_target"], list):
        plt.plot(np.zeros((mat["U_est"].shape[0], mat["U_est"].shape[0])))
    else:
        plt.plot(mat["muscles_target"][count, :], 'r')
    plt.plot(mat["U_est"][i, :])
    plt.title(model.muscleNames()[i].to_string())
    count += 1

if "tau_est" in mat.keys():
    plt.figure("torque")
    plt.plot(mat["tau_est"][:, :].T)

# Compute muscular force at each iteration
X_est = mat["X_est"]
# U_est = mat["U_est"]
# U_est = mat["muscles_target"]
force_est_tmp = np.ndarray((model.nbMuscles(), 1))

plt.figure("Forces")
for i in range(model.nbMuscles()):
    plt.subplot(6, 6, i + 1)
    plt.plot(mat["f_est"][i, :])
    # plt.plot(force_est[i, :])
    plt.title(model.muscleNames()[i].to_string())

delta_t = []
t_ref = []
plt.figure("time")
plt.plot(mat["time"])
plt.figure("delta_time")
for i in range(1, len(mat["time"])):
    delta_t.append(mat["time"][i] - mat["time"][i - 1])
    t_ref.append(1 / 28)
plt.plot(delta_t)
plt.plot(t_ref)

if len(mat["kin_target"].shape) != 2:
    plt.figure("markers")
    # from optim_funct import markers_fun
    # get_markers = markers_fun(model)
    # mark_est = mat["kin_target"].copy()
    # for i in range(mat["kin_target"].shape[2]):
    #     mark_est[:, :, i] = get_markers(mat["X_est"][:2, i])

    for i in range(15):
        plt.plot(mat["kin_target"][:, i, :].T, "-r")
        # plt.plot(mark_est[:, i, :].T, "--b")
plt.show()
