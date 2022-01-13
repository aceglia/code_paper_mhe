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
animate = True

parent = os.path.dirname(os.getcwd())
import scipy.io as sio
# mat = sio.loadmat("/home/amedeo/Documents/programmation/code_paper_mhe/data/data_30_11_21/MVC.mat")
# mat_2 = sio.loadmat("/home/amedeo/Documents/programmation/code_paper_mhe/data/test_18_11_21/gregoire/test_1/test_abd.mat")

file_name = parent + "/" + "results/Jules/Results_MHE_markers_EMG_act_torque_driven_test_20220113-1640"
model = biorbd.Model(parent + "/data/test_09_12_21/Jules/Wu_Shoulder_Model_mod_wt_wrapp_Jules.bioMod")
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
    b.load_experimental_markers(mat["kin_target"])
    # b.load_experimental_markers(c3d)
    b.load_movement(mat["X_est"][:model.nbQ(), :])
    b.exec()


U_est = mat["U_est"]
for i in range(U_est.shape[0]):
    for k in range(U_est.shape[1]):
        if U_est[i, k] == 0:
            U_est[i, k] = 0.01

# U_ref = mat["muscles_target"]
muscle_track_idx = [14, 25, 26,  # PEC
                    13,  # DA
                    15,  # DM
                    21,  # DP
                    23, 24,  # bic
                    28, 29, 30,  # tri
                    10,  # TRAPsup
                    2,  # TRAPmed
                    # 3,  # TRAPinf
                    # 27  # Lat
                    ]
force_est_tmp = np.ndarray((model.nbMuscles(), 1))
from biosiglive.server import Server
# Q, Qdot = Server.kalman_func(markers, model, True)
muscle_force_ref = np.ndarray((model.nbMuscles(), U_est.shape[1]))
count = 0
biorbd_muscle = []
muscles_FLCE = np.ndarray((model.nbMuscles(), U_est.shape[1]))
muscles_FVCE = np.ndarray((model.nbMuscles(), U_est.shape[1]))
muscles_FLPE = np.ndarray((model.nbMuscles(), U_est.shape[1]))
muscles_FLCE_2 = np.ndarray((model.nbMuscles(), U_est.shape[1]))
b11 = 0.815
b21 = 1.055
b31 = 0.162
b41 = 0.063
b12 = 0.433
b22 = 0.717
b32 = -0.03
b42 = 0.200
b13 = 0.100
b23 = 1.000
b33 = 0.354
b43 = 0.0

for i in range(U_est.shape[1]):
    count = 0
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(U_est[k, i])
        biorbd_muscle.append(biorbd.DeGrooteType(model.muscle(k)))
        biorbd_muscle[k].length(model, mat["X_est"][:model.nbQ(), i])
        muscles_FLPE[k, i] = biorbd_muscle[k].FlPE()
        muscles_FLCE[k, i] = biorbd_muscle[k].FlCE(muscles_states[k])
        biorbd_muscle[k].velocity(model, mat["X_est"][:model.nbQ(), i], mat["X_est"][model.nbQ():, i], True)
        muscles_FVCE[k, i] = biorbd_muscle[k].FvCE()
        norm_length = biorbd_muscle[k].length(model, mat["X_est"][:model.nbQ(), i], True) / model.muscle(k).characteristics().optimalLength()
        muscles_FLCE_2[k, i] = b11 * np.exp((-0.5 * ((norm_length - b21) * (norm_length - b21))) \
                             / ((b31 + b41 * norm_length) * (b31 + b41 * norm_length))) \
                             + b12 * np.exp((-0.5 * ((norm_length - b22) * (norm_length - b22))) \
                             / ((b32 + b42 * norm_length) * (b32 + b42 * norm_length))) \
                             + b13 * np.exp((-0.5 * ((norm_length - b23) * (norm_length - b23))) \
                             / ((b33 + b43 * norm_length) * (b33 + b43 * norm_length)))

    muscle_force_ref[:, i] = model.muscleForces(muscles_states,
                                                mat["X_est"][:model.nbQ(), i],
                                                mat["X_est"][model.nbQ():, i]).to_array()

muscle_length = np.zeros((model.nbMuscles(), mat["X_est"].shape[1]))
muscle_tendon_length = np.zeros((model.nbMuscles(), mat["X_est"].shape[1]))
for m in range(model.nbMuscles()):
    for i in range(mat["X_est"].shape[1]):
        muscle_length[m, i] = model.muscle(m).length(model, mat["X_est"][:model.nbQ(), i])
        muscle_tendon_length[m, i] = model.muscle(m).musculoTendonLength(model, mat["X_est"][:model.nbQ(), i])
import matplotlib.pyplot as plt

force_from_act = np.zeros((model.nbMuscles(), mat["f_est"].shape[1]))

for i in range(model.nbMuscles()):
    force_from_act[i, :] = model.muscle(i).characteristics().forceIsoMax()*(
            U_est[i, :]*muscles_FLCE[i, :]*muscles_FVCE[i, :]+muscles_FLPE[i, :])

plt.figure("Muscle force component")
for i in range(model.nbMuscles()):
    plt.subplot(6, 6, i + 1)
    plt.plot(muscles_FLPE[i, :], label="FLPE")
    plt.plot(muscles_FLCE_2[i, :], label="FLPE")
    plt.plot(muscles_FLCE[i, :])
    plt.plot(muscles_FVCE[i, :])

    for k in range(mat["X_est"].shape[1]):
        if force_from_act[i, k] < 0:
            plt.axvline(x=k, alpha=0.2)
        # if mat["f_est"][i, k] < 0:
        #     plt.axvline(x=k, alpha=0.2)
    plt.title(model.muscleNames()[i].to_string())
plt.legend(labels=["FLPE","FLPE2" ,"FLCE", "FVCE"], bbox_to_anchor=(2, -0.50),loc="lower left")

plt.figure("length")
for i in range(model.nbMuscles()):
    plt.subplot(6, 6, i + 1)
    plt.plot(muscle_length[i, :]/model.muscle(i).characteristics().optimalLength(), label="muscle length")
    plt.plot(muscle_tendon_length[i, :], label="muscle tendon length")
    plt.plot(np.repeat(model.muscle(i).characteristics().tendonSlackLength(), muscle_length.shape[1]), 'r', label="tendon slack length")
    # plt.plot(np.repeat(model.muscle(i).characteristics().optimalLength(), muscle_length.shape[1]), 'b', label="optimal length")
    for k in range(mat["X_est"].shape[1]):
        if force_from_act[i, k] < 0:
            plt.axvline(x=k, alpha=0.2)
        # if mat["f_est"][i, k] < 0:
        # # if muscle_force_ref[i, k] < 0:
        #     plt.axvline(x=k, alpha=0.2)
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
    for k in range(mat["X_est"].shape[1]):
        if force_from_act[i, k] < 0:
            plt.axvline(x=k, alpha=0.2)
        # if mat["f_est"][i, f] < 0:
        #     # if muscle_force_ref[i, k] < 0:
        #     plt.axvline(x=f, alpha=0.2)

print(np.mean(mat["sol_freq"][:500]))
print(np.std(mat["sol_freq"][:500]))
# print(mat)
import matplotlib.pyplot as plt

plt.figure("Q")
for i in range(0, int(mat["X_est"].shape[0]/2)):
    plt.subplot(3, 3, i+1)
    plt.plot(mat["X_est"][i, :]*180/np.pi)
    plt.plot(mat["kalman"][i, :]*180/np.pi, '-r')

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
lin = ceil(model.nbMuscles()/col)
count = 0

plt.figure("Muscles tracking")
for i in muscle_track_idx:
# for i in range(model.nbMuscles()):
    plt.subplot(lin, col, count + 1)
    # if isinstance(mat["muscles_target"], list):
    #     plt.plot(np.zeros((1, U_est.shape[1])))
    # else:
    if i in muscle_track_idx:
        plt.plot(mat["muscles_target"][count, :], 'r')
        count += 1
    plt.plot(U_est[i, :])
    plt.title(model.muscleNames()[i].to_string())

# Inverse dynamics
# Choose a position/velocity/acceleration to compute dynamics from
Q = np.zeros((model.nbQ(),))
Qdot = np.zeros((model.nbQ(),))
Qddot = np.zeros((model.nbQ(),))
t = [i - mat["time"][0] for i in mat["time"]]
qddot_mat = np.zeros((int(mat["X_est"].shape[0] / 2), mat["X_est"].shape[1] - 1))
for i in range(1, mat["X_est"].shape[1]):
    qddot_mat[:, i - 1] = (mat["X_est"][:model.nbQ(), i] - mat["X_est"][:model.nbQ(), i - 1]) / (t[i] - t[i-1])
# Proceed with the inverse dynamics
Tau = np.zeros((int(mat["X_est"].shape[0] / 2), mat["X_est"].shape[1] - 1))
for i in range(mat["X_est"].shape[1]):
    Tau[:, i - 1] = model.InverseDynamics(mat["X_est"][:model.nbQ(), i], mat["X_est"][model.nbQ():, i], Qddot).to_array()
if "tau_est" in mat.keys():
    plt.figure("torque")
    for i in range(int(mat["X_est"].shape[0]/2)):
        plt.subplot(3, 3, i + 1)
        plt.plot(mat["tau_est"][i, :])
        plt.plot(Tau[i, :], 'x')

# Compute muscular force at each iteration
X_est = mat["X_est"]

plt.figure("damping")
for i in range(model.nbMuscles()):
    plt.subplot(6, 6, i + 1)
    plt.plot(np.abs(velocity[i, :])/(10 * model.muscle(i).characteristics().optimalLength()*0.1))
    # for k in range(mat["X_est"].shape[1]):
    #     if mat["f_est"][i, k] < 0:
    #         plt.axvline(x=k, alpha=0.2)

plt.figure("Forces")
for i in range(model.nbMuscles()):
    plt.subplot(6, 6, i + 1)
    plt.plot(mat["f_est"][i, :])
    plt.plot(muscle_force_ref[i, :])
    # plt.plot(model.muscle(i).characteristics().forceIsoMax()*(
    #     mat["U_est"][i, :]*muscles_FLCE[i, :]*muscles_FVCE[i, :]
    #     +muscles_FLPE[i, :] + (velocity[i, :]/(10)*0.1)
    #  * np.cos(model.muscle(i).characteristics().pennationAngle())), 'r')
    for k in range(force_from_act.shape[1]):
        if mat["f_est"][i, k] < 0:
            plt.axvline(x=k, alpha=0.2)

    plt.title(model.muscleNames()[i].to_string())
plt.legend(labels=["force_est(casadi)", "force_est(eigen)", "force_from_act(python)"])
delta_t = []
t_ref = []
plt.figure("time")
plt.plot(mat["time"])
plt.figure("delta_time")
for i in range(1, len(mat["time"])):
    delta_t.append(mat["time"][i] - mat["time"][i - 1])
    t_ref.append(1 / 15)
plt.plot(delta_t)
plt.plot(t_ref)

if len(mat["kin_target"].shape) != 2:
    plt.figure("markers")
    # from optim_funct import markers_fun
    # get_markers = markers_fun(model)
    # mark_est = mat["kin_target"].copy()
    # for i in range(mat["kin_target"].shape[2]):
    #     mark_est[:, :, i] = get_markers(mat["X_est"][:2, i])

    for i in range(mat["kin_target"].shape[1]):
        plt.subplot(4, 4, i+1)
        plt.plot(mat["kin_target"][0, i, :].T)#, "-r")
        plt.plot(mat["kin_target"][1, i, :].T)#, "-r")
        plt.plot(mat["kin_target"][2, i, :].T)#, "-r")
        # plt.plot(mark_est[:, i, :].T, "--b")

# static optim
t = [i - mat["time"][0] for i in mat["time"]]
qddot_mat = np.zeros((int(mat["X_est"].shape[0] / 2), mat["X_est"].shape[1] - 1))
for i in range(1, mat["X_est"].shape[1]):
    qddot_mat[:, i - 1] = (mat["X_est"][:model.nbQ(), i] - mat["X_est"][:model.nbQ(), i - 1]) / (t[i] - t[i-1])

Q = biorbd.VecBiorbdGeneralizedCoordinates()
Qdot = biorbd.VecBiorbdGeneralizedVelocity()
Qddot = biorbd.VecBiorbdGeneralizedAcceleration()
Tau = biorbd.VecBiorbdGeneralizedTorque()
for i in range(mat["X_est"].shape[1] - 1):
    Q.append(mat["X_est"][:model.nbQ(), i])
    Qdot.append(mat["X_est"][model.nbQ():, i])
    Qddot.append(qddot_mat[:, i])
    # Tau.append(model.InverseDynamics(Q[i], Qdot[i], Qddot[i]))
    Tau.append(mat["tau_est"][:, i])

# Proceed with the static optimization
# optim = biorbd.StaticOptimization(model, Q, Qdot, Tau)
# optim.run()
# muscleActivationsPerFrame = optim.finalSolution()
activation_static_optim = np.zeros((model.nbMuscles(), mat["X_est"].shape[1] - 1))
# Print them to the console
# for i, activations in enumerate(muscleActivationsPerFrame):
#     activation_static_optim[:, i] = activations.to_array()

plt.figure("Muscles")
# for i in muscle_track_idx:
for i in range(model.nbMuscles()):
    plt.subplot(lin, col, i + 1)
    plt.plot(U_est[i, :])
    plt.plot(activation_static_optim[i, :])
    plt.title(model.muscleNames()[i].to_string())

plt.show()

