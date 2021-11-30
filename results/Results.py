import numpy as np
from biosiglive.data_processing import read_data
from biosiglive.server import Server
import biorbd
import os
from pyomeca import Markers
import matplotlib.pyplot as plt
use_torque = True
animate = True

parent = os.path.dirname(os.getcwd())
file_name = parent + "/" + "results/results_20211130/Results_MHE_q_EMG_act_torque_driven_test_20211130-1330"
model = biorbd.Model(parent + "/data/data_30_11_21/Wu_Shoulder_Model_mod_wt_wrapp_remi.bioMod")
c3d = parent + "/data/data_09_2021/abd.c3d"
mat = read_data(file_name)

nbGT = model.nbGeneralizedTorque() if use_torque is True else 0

# Recons markers with kalman:
# q_recons, q_dot = Server.kalman_func(mat["kin_target"][:, :, :], model=model)

# Markers from c3d
markers = Markers.from_c3d(c3d).values
markers = markers * 0.001

if animate is True:
    import bioviz
    b = bioviz.Viz(loaded_model=model)
    # b.load_experimental_markers(mat["kin_target"])
    # b.load_experimental_markers(c3d)
    b.load_movement(mat["X_est"][:model.nbQ(), :])
    b.exec()


U_est = mat["U_est"]
U_ref = mat["muscles_target"]
muscle_track_idx = [13, 15, 21, 10, 1, 2, 27, 14, 25, 26, 23, 24, 28, 29, 30]
force_est_tmp = np.ndarray((model.nbMuscles(), 1))
from biosiglive.server import Server
Q, Qdot = Server.kalman_func(markers, model, True)
muscle_force_ref = np.ndarray((model.nbMuscles(), U_est.shape[1]))
count = 0
biorbd_muscle = []
muscles_FLCE = np.ndarray((model.nbMuscles(), U_est.shape[1]))
muscles_FVCE = np.ndarray((model.nbMuscles(), U_est.shape[1]))
muscles_FLPE = np.ndarray((model.nbMuscles(), U_est.shape[1]))
for i in range(U_est.shape[1]):
    count = 0
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(U_est[k, i])
        biorbd_muscle.append(biorbd.HillThelenType(model.muscle(k)))
        biorbd_muscle[k].length(model, mat["X_est"][:model.nbQ(), i])
        muscles_FLCE[k, i] = biorbd_muscle[k].FlCE(muscles_states[k])
        biorbd_muscle[k].velocity(model, mat["X_est"][:model.nbQ(), i], mat["X_est"][model.nbQ():, i], True)
        muscles_FVCE[k, i] = biorbd_muscle[k].FvCE()
        muscles_FLPE[k, i] = biorbd_muscle[k].FlPE()

    muscle_force_ref[:, i] = model.muscleForces(muscles_states,
                                                mat["X_est"][:model.nbQ(), i],
                                                mat["X_est"][model.nbQ():, i]).to_array()

muscle_length = np.zeros((model.nbMuscles(), mat["X_est"].shape[1]))
muscle_tendon_length = np.zeros((model.nbMuscles(), mat["X_est"].shape[1]))
for m in range(model.nbMuscles()):
    for i in range(mat["X_est"].shape[1]):
        muscle_length[m, i] = model.muscle(0).length(model, mat["X_est"][:model.nbQ(), i])
        muscle_tendon_length[m, i] = model.muscle(0).musculoTendonLength(model, mat["X_est"][:model.nbQ(), i])
import matplotlib.pyplot as plt
plt.figure("Muscle force component")
for i in range(model.nbMuscles()):
    plt.subplot(6, 6, i + 1)
    plt.plot(muscles_FLPE[i, :], label="FLPE")
    plt.plot(muscles_FLCE[i, :])
    plt.plot(muscles_FVCE[i, :])

    for k in range(mat["X_est"].shape[1]):
        if mat["f_est"][i, k] < 0:
            plt.axvline(x=k, alpha=0.2)
    plt.title(model.muscleNames()[i].to_string())
plt.legend(labels=["FLPE", "FLCE", "FVCE"], bbox_to_anchor=(2, -0.50),loc="lower left")

plt.figure("length")
for i in range(model.nbMuscles()):
    plt.subplot(6, 6, i + 1)
    # plt.plot(muscle_length[i, :], label="muscle length")
    plt.plot(muscle_tendon_length[i, :], label="muscle tendon length")
    # plt.plot(muscles_FLPE[i, :], label="FLPE")
    # plt.plot(muscles_FLCE[i, :])
    # plt.plot(mat["U_est"][i, :], label="muscle_length")
    plt.plot(np.repeat(model.muscle(i).characteristics().tendonSlackLength(), muscle_length.shape[1]), 'r', label="tendon slack length")
    # plt.plot(np.repeat(model.muscle(i).characteristics().optimalLength(), muscle_length.shape[1]), 'b', label="optimal length")
    for k in range(mat["X_est"].shape[1]):
        if mat["f_est"][i, k] < 0:
        # if muscle_force_ref[i, k] < 0:
            plt.axvline(x=k, alpha=0.2)
    # plt.plot(force_est[i, :])
    plt.title(model.muscleNames()[i].to_string())
plt.legend(labels=["muscle tendon length", "tendon slack length"], bbox_to_anchor=(2, -0.50),loc="lower left")
plt.figure("muscle velocity")
velocity = np.zeros((model.nbMuscles(), mat["X_est"].shape[1]))
for i in range(model.nbMuscles()):
    plt.subplot(6, 6, i + 1)
    for f in range(mat["X_est"].shape[1]):
        velocity[i, f] = model.muscle(i).velocity(model, mat["X_est"][:model.nbQ(), f], mat["X_est"][model.nbQ():, f])
    plt.plot(velocity[i, :])
    for f in range(mat["X_est"].shape[1]):
        if mat["f_est"][i, f] < 0:
            # if muscle_force_ref[i, k] < 0:
            plt.axvline(x=f, alpha=0.2)

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

plt.figure("Qdot")
for i in range(0, int(mat["X_est"].shape[0]/2)):
    plt.plot(mat["X_est"][int(mat["X_est"].shape[0]/2)+i, :])
    # if len(mat["kin_target"].shape) == 2:
    #     plt.plot(mat["kin_target"][i, :]*180/np.pi, '0')
    # plt.plot(mat["kalman"][int(mat["X_est"].shape[0]/2)+i, :], 'x')
    # plt.plot(q_recons[i, :] * 180 / np.pi, 'x')

# plt.figure("markers")
# for i in range(3):
#     plt.plot(mat["markers_target"][i, :, :].T)
from math import ceil
col = 4
lin = ceil(len(muscle_track_idx)/col)
count = 0
plt.figure("Muscles")
for i in muscle_track_idx:
    plt.subplot(lin, col, count + 1)
    if isinstance(mat["muscles_target"], list):
        plt.plot(np.zeros((1, mat["U_est"].shape[1])))
    else:
        plt.plot(mat["muscles_target"][count, :], 'r')
    plt.plot(mat["U_est"][i, :])
    plt.title(model.muscleNames()[i].to_string())
    count += 1

# Inverse dynamics
# Choose a position/velocity/acceleration to compute dynamics from
Q = np.zeros((model.nbQ(),))
Qdot = np.zeros((model.nbQ(),))
Qddot = np.zeros((model.nbQ(),))

# Proceed with the inverse dynamics
Tau = model.InverseDynamics(Q, Qdot, Qddot)
if "tau_est" in mat.keys():
    plt.figure("torque")
    # plt.plot(mat["tau_est"][:, :].T)
    plt.plot(Tau.to_array().T, 'x')

# Compute muscular force at each iteration
X_est = mat["X_est"]


plt.figure("Forces")
for i in range(model.nbMuscles()):
    plt.subplot(6, 6, i + 1)
    plt.plot(mat["f_est"][i, :])
    plt.plot(muscle_force_ref[i, :])
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
