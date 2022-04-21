import numpy as np
import scipy.interpolate

from biosiglive.data_processing import read_data
from biosiglive.server import Server

try:
    import biorbd
except:
    import biorbd_casadi as biorbd
import os

from pyomeca import Markers
import bioviz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

use_torque = True
animate = False
import seaborn
import glob

# --- RMSE --- #
def rmse(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).mean())


def std(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).std())


parent = os.path.dirname(os.getcwd())
subject = ["subject_1"]  # , "Mathis"]#, "subject_2"]
# trial = ["cycl_cocon_wt_damping_wt_torque_result"]#, "Results_mhe_markers_EMG_act_torque_driven_20220214-1819"]#, "test_abd_full"]
trial = ["cycl_cocon_result"]  # , "abd_SQP_result"]
result_mat = []
nb_full_node = None
models = []
for subj in subject:
    result_dir = f"{subj}/"
    for i in range(len(trial)):
        file = result_dir + f"{trial[i]}"
        result_mat.append(read_data(file))
        if "full" in trial[i]:
            nb_full_node = result_mat[i]["U_est"].shape[1]
        else:
            nb_mhe = result_mat[i]["U_est"].shape[1]
    models.append(biorbd.Model(f"{subj}/model_{subj}_scaled.bioMod"))

onset_limit = 0.05
muscle_track_idx = [
    14,
    25,
    26,  # PEC
    13,  # DA
    15,  # DM
    21,  # DP
    23,
    24,  # bic
    28,
    29,
    30,  # tri
    10,  # TRAPsup
    2,  # TRAPmed
    3,  # TRAPinf
    27,  # Lat
]
# ref = read_data("/home/amedeo/Documents/programmation/data_article/subject_1/abd_cocon")
# kalman = ref["kalman"]
interset_muscle = [0, 1, 4, 5, 6, 7, 8, 9, 12, 14, 17, 19, 22, 25]
# interset_muscle = []
if not nb_full_node:
    tf = nb_mhe / 35
else:
    tf = nb_full_node / 100
t = np.linspace(0, tf, nb_mhe)
for i in range(len(result_mat)):
    if "full" in trial[i]:
        x = np.linspace(0, tf, nb_full_node - 7)
        x_new = t
        # interpolation full data
        # muscles
        f_mus = interp1d(x, result_mat[i]["U_est"][:, 7:])
        full_muscle_est = f_mus(x_new)

        # Q
        f_Q = interp1d(x, result_mat[i]["X_est"][:, 7:])
        full_Q_est = f_Q(x_new)

        f_Q_int = interp1d(x, result_mat[i]["q_int"][:, 7:])
        full_Q_int = f_Q_int(x_new)

        f_kalman = interp1d(x, result_mat[i]["kalman"][:, 7:])
        full_kalman = f_kalman(x_new)

        # f_ref_kalman = interp1d(x, kalman[:, :-3])
        # ref_kalman = f_ref_kalman(x_new)

# b = bioviz.Viz(model_path=f"{subject[0]}/model_{subject[0]}_scaled.bioMod",
#                show_muscles=True,
#                show_floor=False,
#                show_local_ref_frame=False,
#                show_global_ref_frame=False,
#                show_gravity_vector=True,
#                )
#
# b.load_movement(result_mat[0]["X_est"][:models[0].nbQ(), :])
# b.load_experimental_markers(result_mat[0]["kin_target"][:, :, :])
# b.exec()
# for i in range(result_mat[0]["kalman"].shape[1] - 4000):
#     dic_to_save = {"kalman": result_mat[0]["kalman"][:, 4000 + i:],
#                    "markers":
#                    }

# Seaborn stuff
seaborn.color_palette()

grid_line_style = "--"
co_con_ref_style = ""
co_con_est_style = ""
ref_style = ""
est_style = ""

est_style_list = [est_style, co_con_est_style]
ref_style_list = [ref_style, co_con_ref_style]
rmse_q_full = []
rmse_q = []
plt.figure("Q")
for s in range(len(trial)):
    for i in range(models[0].nbQ()):
        plt.subplot(3, 5, i + 1)
        if "full" in trial[s]:
            plt.plot(t, full_Q_est[i, :] * 180 / np.pi, "-g", label="full_est")
            plt.plot(t, full_kalman[i, :] * 180 / np.pi, "--g", label="full_ref")
            # rmse_tmp = rmse(result_mat[s]["X_est"][i, :], result_mat[s]["kalman"][i, :])
            #
            # rmse_q_full.append(rmse_tmp * 57.3)
        else:
            plt.plot(result_mat[s]["X_est"][i, :] * 180 / np.pi, label=trial[s])
            plt.plot(result_mat[s]["kalman"][i, :] * 180 / np.pi, "-r", label="mhe_ref")
            # plt.plot(t[:], result_mat[s]["q_int"][i, :] * 180 / np.pi, "--b", label="mhe_int")
            # rmse_tmp = rmse(result_mat[s]["X_est"][i, :], result_mat[s]["kalman"][i, :])
            # rmse_q.append(rmse_tmp * 57.3)
        if s == 0:
            plt.title(models[0].nameDof()[i].to_string())
        plt.tight_layout()
plt.legend()

plt.figure("Q_dot")
for s in range(len(trial)):
    for i in range(models[0].nbQ(), models[0].nbQ() * 2):
        plt.subplot(3, 5, i - models[0].nbQ() + 1)
        if "full" in trial[s]:
            plt.plot(full_Q_est[i, :] * 180 / np.pi, "-g", label="full_est")
            # plt.plot(t, full_kalman[i, :] * 180 / np.pi, "--g", label="full_ref")
            # rmse_tmp = rmse(result_mat[s]["X_est"][i, :], result_mat[s]["kalman"][i, :])
            #
            # rmse_q_full.append(rmse_tmp * 57.3)
        else:
            plt.plot(result_mat[s]["X_est"][i, :] * 180 / np.pi, label=trial[s])
            # plt.plot(t[:], result_mat[s]["kalman"][i, :] * 180 / np.pi, "-r", label="mhe_ref")
            # plt.plot(t[:], result_mat[s]["q_int"][i, :] * 180 / np.pi, "--b", label="mhe_int")
            # rmse_tmp = rmse(result_mat[s]["X_est"][i, :], result_mat[s]["kalman"][i, :])
            # rmse_q.append(rmse_tmp * 57.3)
        # if s == 0:
        #     plt.title(models[0].nameDof()[i].to_string())
        plt.tight_layout()
plt.legend()
# plt.figure("torque")
# for s in range(len(trial)):
#     for i in range(models[0].nbQ()):
#         plt.subplot(3, 3, i + 1)
#         if "full" in trial[s]:
#             plt.plot(t[s],result_mat[s]["tau_est"][:, ::ratio][i, :], '--g')
#         else:
#             plt.plot(t[s],result_mat[s]["tau_est"][i, :], est_style_list[s])
#         if s == 0:
#             plt.title(models[0].nameDof()[i].to_string())
#         plt.tight_layout()


rmse_markers = []
rmse_markers_full = []
rmse_markers_full_ref = []
for s in range(len(trial)):
    plt.figure("markers")
    if "full" not in trial[s]:
        markers = np.ndarray((3, result_mat[s]["kin_target"].shape[1], result_mat[s]["kin_target"].shape[2]))
        for i in range(result_mat[s]["kin_target"].shape[2]):
            markers[:, :, i] = np.array([mark.to_array() for mark in models[0].markers(result_mat[s]["X_est"][:, i])]).T
    if "full" in trial[s]:
        markers_full = np.ndarray((3, result_mat[s]["kin_target"].shape[1], full_Q_est.shape[1]))
        for i in range(full_Q_est.shape[1]):
            markers_full[:, :, i] = np.array([mark.to_array() for mark in models[0].markers(full_Q_est[:, i])]).T
        markers_full_ref = np.ndarray((3, result_mat[s]["kin_target"].shape[1], full_kalman.shape[1]))
        for i in range(full_kalman.shape[1]):
            markers_full_ref[:, :, i] = np.array([mark.to_array() for mark in models[0].markers(full_kalman[:, i])]).T
        markers_full_tot = np.ndarray((3, result_mat[s]["kin_target"].shape[1], result_mat[s]["X_est"].shape[1]))
        for i in range(result_mat[s]["X_est"].shape[1]):
            markers_full_tot[:, :, i] = np.array(
                [mark.to_array() for mark in models[0].markers(result_mat[s]["X_est"][:, i])]
            ).T
        markers_full_ref_tot = np.ndarray((3, result_mat[s]["kin_target"].shape[1], result_mat[s]["X_est"].shape[1]))
        for i in range(result_mat[s]["X_est"].shape[1]):
            markers_full_ref_tot[:, :, i] = np.array(
                [mark.to_array() for mark in models[0].markers(result_mat[s]["kalman"][:, i])]
            ).T

    for i in range(result_mat[s]["kin_target"].shape[1]):
        plt.subplot(4, 4, i + 1)
        if i == 0:
            if "full" in trial[s]:
                plt.plot(t, markers_full[0, i, :].T, "-g", alpha=0.8, label="Forward kin full")
                plt.plot(t, markers_full[1, i, :].T, "-g", alpha=0.8)
                plt.plot(t, markers_full[2, i, :].T, "-g", alpha=0.8)
                # for j in range(result_mat[0]["kin_target"].shape[0]):
                #     rmse_tmp = rmse(markers_full_tot[j, i, :], result_mat[s]['kin_target'][j, i, :])
                #     rmse_markers_full.append(rmse_tmp * 1000)
                # for j in range(result_mat[0]["kin_target"].shape[0]):
                #     rmse_tmp = rmse(markers_full_ref_tot[j, i, :], result_mat[s]['kin_target'][j, i, :])
                #     rmse_markers_full_ref.append(rmse_tmp * 1000)
            else:
                plt.plot(result_mat[s]["kin_target"][0, i, :].T, "--r", label="kalman")
                plt.plot(result_mat[s]["kin_target"][1, i, :].T, "--r")
                plt.plot(result_mat[s]["kin_target"][2, i, :].T, "--r")
                plt.plot(markers[0, i, :].T, alpha=0.8, label=trial[s])
                plt.plot(markers[1, i, :].T, alpha=0.8)
                plt.plot(markers[2, i, :].T, alpha=0.8)
                # for j in range(result_mat[0]["kin_target"].shape[0]):
                #     rmse_tmp = rmse(markers[j, i, :], result_mat[s]['kin_target'][j, i, :])
                #     rmse_markers.append(rmse_tmp * 1000)

        else:
            if "full" in trial[s]:
                plt.plot(t, markers_full[0, i, :].T, "-g", alpha=0.8)
                plt.plot(t, markers_full[1, i, :].T, "-g", alpha=0.8)
                plt.plot(t, markers_full[2, i, :].T, "-g", alpha=0.8)
                # for j in range(result_mat[0]["kin_target"].shape[0]):
                #     rmse_tmp = rmse(markers_full_tot[j, i, :], result_mat[s]['kin_target'][j, i, :])
                #     rmse_markers_full.append(rmse_tmp * 1000)
                # for j in range(result_mat[0]["kin_target"].shape[0]):
                #     rmse_tmp = rmse(markers_full_ref_tot[j, i, :], result_mat[s]['kin_target'][j, i, :])
                #     rmse_markers_full_ref.append(rmse_tmp * 1000)
            else:
                plt.plot(result_mat[s]["kin_target"][0, i, :].T, "--r")
                plt.plot(result_mat[s]["kin_target"][1, i, :].T, "--r")
                plt.plot(result_mat[s]["kin_target"][2, i, :].T, "--r")
                plt.plot(markers[0, i, :].T, alpha=0.8)
                plt.plot(markers[1, i, :].T, alpha=0.8)
                plt.plot(markers[2, i, :].T, alpha=0.8)
                for j in range(result_mat[0]["kin_target"].shape[0]):
                    rmse_tmp = rmse(markers[j, i, :], result_mat[s]["kin_target"][j, i, :])
                    rmse_markers.append(rmse_tmp * 1000)

plt.legend()
#
print(f"RMSE markers positions :{np.round(np.mean(rmse_markers), 3)} mm")
print(f"RMSE markers positions full:{np.round(np.mean(rmse_markers_full), 3)} mm")
print(f"RMSE markers positions full ref:{np.round(np.mean(rmse_markers_full_ref), 3)} mm")

print(f"RMSE Q :{np.mean(rmse_q)} °")
print(f"RMSE Q full :{np.mean(rmse_q_full)} °")


plt.figure("Estimated activation and EMG signals ")
plt.gcf().subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.95, wspace=0.18, hspace=0.35)
count = 0
rmse_emg = []
rmse_emg_full = []
for s in range(len(trial)):
    # for i in muscle_track_idx:
    count = 0
    c = 0
    for i in range(models[0].nbMuscles()):
        fig = plt.subplot(5, 7, count + 1)
        # if i not in interset_muscle:
        if i in muscle_track_idx:
            idx = muscle_track_idx.index(i)
            if not "full" in trial[s] and not "wt" in trial[s]:
                plt.plot(result_mat[s]["muscles_target"][idx, :], "-r")
            if "wt" in trial[s]:
                if idx == result_mat[s]["muscles_target"].shape[0]:
                    pass
                else:
                    plt.plot(result_mat[s]["muscles_target"][idx, :], "-r")
        if count == 0:
            if not "full" in trial[s]:
                plt.plot(t, result_mat[s]["U_est"][i, :], label=trial[s])
            else:
                plt.plot(t, full_muscle_est[i, :], "-g", label="full_activation")
        else:
            if not "full" in trial[s]:
                plt.plot(t, result_mat[s]["U_est"][i, :])
            else:
                plt.plot(t, full_muscle_est[i, :], "-g")
        if count not in [16, 17, 18, 19, 20]:
            fig.set_xticklabels([])
        if count not in [0, 4, 8, 12, 16, 25, 30]:
            fig.set_ylim(0, 1)
            fig.set_yticklabels([])
        else:
            fig.set_ylim(0, 1)
            # fig.set_ylim(0, 1)

        if s == 0:
            plt.title(models[0].muscleNames()[i].to_string())
        plt.grid(True)
        count += 1
# legend
plt.legend(labels=[f"Estimated activation", "EMG"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.figure("Forces")
plt.gcf().subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.95, wspace=0.18, hspace=0.35)
count = 0
rmse_emg = []
rmse_emg_full = []
for s in range(len(trial)):
    # for i in muscle_track_idx:
    count = 0
    c = 0
    for i in range(models[0].nbMuscles()):
        fig = plt.subplot(5, 7, count + 1)
        if i not in interset_muscle:
            f_iso = models[0].muscle(i).characteristics().forceIsoMax()
            if i in muscle_track_idx:
                idx = muscle_track_idx.index(i)
            if count == 0:
                if not "full" in trial[s]:
                    plt.plot(result_mat[s]["f_est"][i, :] / f_iso * 100, label=trial[s])
            else:
                if not "full" in trial[s]:
                    plt.plot(result_mat[s]["f_est"][i, :] / f_iso * 100)
            if count not in [16, 17, 18, 19, 20]:
                fig.set_xticklabels([])
            if count not in [0, 4, 8, 12, 16, 25, 30]:
                fig.set_yticklabels([])
                fig.set_ylim(0, 50)
            else:
                fig.set_ylim(0, 50)

            if s == 0:
                plt.title(models[0].muscleNames()[i].to_string())
            plt.grid(True)
            count += 1
rmse_int = []
# for i in range(len(trial)):
#     if "full" not in trial[i]:
#         for j in range(models[0].nbQ()):
#             rmse_tmp = rmse(result_mat[i]["q_int"][j, :-nb_mhe], result_mat[i]["q"][j, :])
#             rmse_int.append(rmse_tmp)
# print(f"rmse single shoot / subproblem :{np.mean(rmse_int)*37.5}°")

plt.legend()
# plt.tight_layout()
# plt.savefig("subject_1/results/abd_muscle.png", format="png")

# plt.figure("Muscles")
# for s in range(len(subject)):
#     for i in range(result_mat[s]["emg_proc"].shape[0]):
#         plt.subplot(3, 4, i + 1)
#         plt.plot(result_mat[s]["emg_proc"][i, 7000:], '-r')
#         if s == 0:
#             plt.title(models[0].muscleNames()[i].to_string())
#         plt.tight_layout()
plt.show()
