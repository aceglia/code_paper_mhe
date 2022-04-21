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
subjects = ["subject_1", "subject_2"]  # , "Mathis"]#, "subject_2"]
trials = ["abd_result", "flex_result", "cycl_result", "abd_cocon_result", "flex_cocon_result", "cycl_cocon_result"]
rmse_variables = ["markers", "q", "emg"]
result_mat = np.zeros((len(subjects), len(trials), 5))
rmse_mat = np.zeros((len(subjects), len(rmse_variables)))

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
# not_interest_muscles = [0, 1, 4, 5, 6, 7, 8, 9, 12, 19, 20, 22]

rmse_variables = ["markers", "q", "emg"]

rmse_mhe = np.zeros((len(subjects), len(trials), len(rmse_variables)))
std_mhe = np.zeros((len(subjects), len(trials), len(rmse_variables)))

for s, subject in enumerate(subjects):
    model = biorbd.Model(f"{subject}/model_{subject}_scaled.bioMod")
    if subject == "subject_2":
        trials = trials[0:3]
    for t, trial in enumerate(trials):
        file = f"{subject}/" + f"{trial}"
        data = read_data(file)
        markers_ref = data["kin_target"]  # markers target
        q = data["X_est"]  # q
        act = data["U_est"]  # EMG
        emg = data["muscles_target"]  # EMG ref
        kalman = data["kalman"]  # kalman
        for v, variable in enumerate(rmse_variables):
            rmse_tmp = []
            if variable == "q":
                for i in range(model.nbQ()):
                    rmse_tmp.append(rmse(q[i, :], kalman[i, :]) * (180 / np.pi))

            elif variable == "markers":
                markers_tmp = np.ndarray((3, markers_ref.shape[1], markers_ref.shape[2]))
                for i in range(markers_ref.shape[2]):
                    markers_tmp[:, :, i] = np.array([mark.to_array() for mark in model.markers(q[:, i])]).T
                for i in range(markers_ref.shape[1]):
                    for j in range(markers_ref.shape[0]):
                        rmse_tmp.append(rmse(markers_tmp[j, i, :], markers_ref[j, i, :]) * 1000)

            elif variable == "emg":
                for i in range(model.nbMuscles()):
                    if i in muscle_track_idx:
                        idx = muscle_track_idx.index(i)
                        rmse_tmp.append(rmse(act[i, :], emg[idx, :]) * 100)
            print(f"RMSE on {variable} for {trial} for subject {subject}: {np.mean(rmse_tmp)} +/- {np.std(rmse_tmp)}")
            rmse_mhe[s, t, v] = np.round(np.mean(rmse_tmp), 2)
            std_mhe[s, t, v] = np.round(np.std(rmse_tmp), 2)

print(
    "\ begin{tabular}[c]{cc|cc|cc|cc|cc|cc|cc}\n"
    "& & \multicolumn{4}{c|}{Markers (mm)}&\n"
    "\multicolumn{4}{c|}{Joint angles (°)} &\n"
    "\multicolumn{4}{c}{Activations($\%$)} \ \ \n"
    "Movement & Additional contraction & \n"
    "\multicolumn{2}{c}{"
    f"{subjects[0]}"
    "} & \multicolumn{2}{c|}{"
    f"{subjects[1]}"
    "}&\n \multicolumn{2}{c}{"
    f"{subjects[0]}"
    "} "
    "& \multicolumn{2}{c|}{"
    f"{subjects[1]}"
    "}&\n \multicolumn{2}{c}{"
    f"{subjects[0]}"
    "} & \multicolumn{2}{c}{"
    f"{subjects[1]}"
    "} \ \ \n"
    "& & mean & SD & mean & SD & mean & SD & mean & SD & mean & SD & mean & SD \ \ \n"
    "\hline\n"
    f"Abduction & without & ${rmse_mhe[0,0,0]}$ & ${std_mhe[0,0,0]}$ & ${rmse_mhe[1,0,0]}$ & ${std_mhe[1,0,0]}$ & "
    f"${rmse_mhe[0,0,1]}$ & ${std_mhe[0,0,1]}$ & ${rmse_mhe[1,0,1]}$ & ${std_mhe[1,0,1]}$ &"
    f" ${rmse_mhe[0,0,2]}$ & ${std_mhe[0,0,2]}$ & ${rmse_mhe[1,0,2]}$ & ${std_mhe[1,0,2]}$\ \ \n"
    f"         & with & ${rmse_mhe[0,3,0]}$ & ${std_mhe[0,3,0]}$ & - & - & "
    f"${rmse_mhe[0,3,1]}$ & ${std_mhe[0,3,1]}$ & - & - &"
    f" ${rmse_mhe[0,3,2]}$ & ${std_mhe[0,3,2]}$ & - & - \ \ \n"
    "        \hline\n"
    f"Flexion & without & ${rmse_mhe[0,1,0]}$ & ${std_mhe[0,1,0]}$ & ${rmse_mhe[1,1,0]}$ & ${std_mhe[1,1,0]}$ & "
    f"${rmse_mhe[0,1,1]}$ & ${std_mhe[0,1,1]}$ & ${rmse_mhe[1,1,1]}$ & ${std_mhe[1,1,1]}$ &"
    f" ${rmse_mhe[0,1,2]}$ & ${std_mhe[0,1,2]}$ & ${rmse_mhe[1,1,2]}$ & ${std_mhe[1,1,2]}$\ \ \n"
    f"         & with & ${rmse_mhe[0,4,0]}$ & ${std_mhe[0,4,0]}$ & - & - & "
    f"${rmse_mhe[0,4,1]}$ & ${std_mhe[0,4,1]}$ & - & - &"
    f" ${rmse_mhe[0,4,2]}$ & ${std_mhe[0,4,2]}$ & - & - \ \ \n"
    "        \hline\n"
    f"Hand-cycling & without & ${rmse_mhe[0,2,0]}$ & ${std_mhe[0,2,0]}$ & ${rmse_mhe[1,2,0]}$ & ${std_mhe[1,2,0]}$ & "
    f"${rmse_mhe[0,2,1]}$ & ${std_mhe[0,2,1]}$ & ${rmse_mhe[1,2,1]}$ & ${std_mhe[1,2,1]}$ &"
    f" ${rmse_mhe[0,2,2]}$ & ${std_mhe[0,2,2]}$ & ${rmse_mhe[1,2,2]}$ & ${std_mhe[1,2,2]}$\ \ \n"
    f"         & with & ${rmse_mhe[0,5,0]}$ & ${std_mhe[0,5,0]}$ & - & - & "
    f"${rmse_mhe[0,5,1]}$ & ${std_mhe[0,5,1]}$ & - & - &"
    f" ${rmse_mhe[0,5,2]}$ & ${std_mhe[0,5,2]}$ & - & - \ \ \n"
    "        \hline\n"
    f"Total & & ${np.round(np.mean(rmse_mhe[0,:,0]), 2)}$ & ${np.round(np.mean(std_mhe[0,:,0]), 2)}$ & ${np.round(np.mean(rmse_mhe[1,:3,0]), 2)}$ & ${np.round(np.mean(std_mhe[1,:3,0]), 2)}$ & "
    f"${np.round(np.mean(rmse_mhe[0,:,1]), 2)}$ & ${np.round(np.mean(std_mhe[0,:,1]), 2)}$ & ${np.round(np.mean(rmse_mhe[1,:3,1]), 2)}$ & ${np.round(np.mean(std_mhe[1,:3,1]), 2)}$ &"
    f" ${np.round(np.mean(rmse_mhe[0,:,2]), 2)}$ & ${np.round(np.mean(std_mhe[0,:,2]), 2)}$ & ${np.round(np.mean(rmse_mhe[1,:3,2]), 2)}$ & ${np.round(np.mean(std_mhe[1,:3,2]), 2)}$\ \ \n"
    "\end{tabular}\n"
)
