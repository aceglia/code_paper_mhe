import numpy as np
from biosiglive.data_processing import read_data

try:
    import biorbd
except:
    import biorbd_casadi as biorbd
import os

# --- RMSE --- #
def rmse(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).mean())


def std(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).std())


parent = os.path.dirname(os.getcwd())
subject = "subject_1"  # , "Mathis"]#, "subject_2"]
trials = ["abd_result", "abd_SQP_result"]
rmse_variables = ["markers", "q", "emg"]
rmse_mat = np.zeros((1, len(rmse_variables)))

rmse_variables = ["markers", "q", "emg"]

rmse_mhe = np.zeros((1, len(trials), len(rmse_variables)))
std_mhe = np.zeros((1, len(trials), len(rmse_variables)))

model = biorbd.Model(f"{subject}/model_{subject}_scaled.bioMod")
result_mat = []
nb_full_node = None
q, q_sqp = [], []

for i in range(len(trials)):
    file = f"subject_1/{trials[i]}"
    result_mat.append(read_data(file))
    data = read_data(file)
    if "SQP" in trials[i]:
        q_sqp = data["X_est"][: model.nbQ(), :]  # q
        act_sqp = data["U_est"]
        markers_sqp = np.ndarray((3, data["kin_target"].shape[1], data["kin_target"].shape[2]))
        for i in range(data["kin_target"].shape[2]):
            markers_sqp[:, :, i] = np.array([mark.to_array() for mark in model.markers(data["X_est"][:, i])]).T
        freq_sqp = data["sol_freq"][1:]

    else:
        q = data["X_est"][: model.nbQ(), :]  # q
        act = data["U_est"]  # EMG
        markers = np.ndarray((3, data["kin_target"].shape[1], data["kin_target"].shape[2]))
        for i in range(data["kin_target"].shape[2]):
            markers[:, :, i] = np.array([mark.to_array() for mark in model.markers(data["X_est"][:, i])]).T
        freq = data["sol_freq"][1:]

# --- RMSE --- #
rmse_q = np.round(rmse(q, q_sqp) * (180 / np.pi), 2)
std_q = np.round(std(q, q_sqp) * (180 / np.pi), 2)
rmse_emg = np.round(rmse(act, act_sqp) * 100, 2)
std_emg = np.round(std(act, act_sqp) * 100, 2)
rmse_tmp = []
std_tmp = []
for i in range(markers_sqp.shape[1]):
    for j in range(markers_sqp.shape[0]):
        rmse_tmp.append(rmse(markers[j, i, :], markers_sqp[j, i, :]) * 1000)
        std_tmp.append(std(markers[j, i, :], markers_sqp[j, i, :]) * 1000)
rmse_markers = np.round(np.mean(rmse_tmp), 2)
std_markers = np.round(np.mean(std_tmp), 2)
mean_freq_sqp = np.round(np.mean(freq_sqp), 2)
std_freq_sqp = np.round(np.std(freq_sqp), 2)
mean_freq = np.round(np.mean(freq), 2)
std_freq = np.round(np.std(freq), 2)


print(
    "\ begin{tabular}[c]{l|cc}\n"
    "& \multicolumn{2}{c}{RMSE} \ \ \n"
    f"& mean & SD \ \ \n"
    f"\hline \n"
    f"Markers (mm) & ${rmse_markers}$ & ${std_markers}$\ \ \n"
    f"Joint angles (°) & ${rmse_q}$ & ${std_q}$\ \ \n"
    f"Muscle activations ($\%$) & ${rmse_emg}$ & ${std_emg}$ \ \ \n"
    f"\hline "
    f"\hline "
    "& \multicolumn{2}{c}{Working frequency (Hz)} \ \ \n"
    "& mean & SD \ \ \n"
    f"\hline \n"
    f"SQP & ${mean_freq_sqp}$ & ${std_freq_sqp}$\ \ \n"
    f"SQP$\_$RTI & ${mean_freq}$ & ${std_freq}$ \ \ \n"
)
