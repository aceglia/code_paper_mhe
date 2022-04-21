import seaborn as sns
import numpy as np
from biosiglive.data_processing import read_data
import os
from scipy.interpolate import interp1d
import biorbd
import matplotlib.pyplot as plt
import pandas as pd


parent = os.path.dirname(os.getcwd())
subject = ["subject_1"]
trial = ["full_abd_1_rep_result", "abd_1_rep_result"]
result_mat = []
models = []
for subj in subject:
    result_dir = f"{subj}/"
    for i in range(len(trial)):
        file = result_dir + f"{trial[i]}"
        result_mat.append(read_data(file))
        if "full" in trial[i]:
            nb_full_node = result_mat[i]["U_est"].shape[1]
            emg_full = result_mat[i]["U_est"]
        else:
            nb_mhe = result_mat[i]["U_est"].shape[1]
            emg_mhe = result_mat[i]["U_est"][:, :]
            emg_ref = result_mat[i]["muscles_target"][:, :]
    models.append(biorbd.Model(f"{subj}/Wu_Shoulder_Model_mod_wt_wrapp_{subj}_scaled.bioMod"))

onset_limit = 0.08
tf = nb_full_node / 100
t = np.linspace(0, tf, nb_mhe)
for i in range(len(result_mat)):
    if "full" in trial[i]:
        x = np.linspace(0, tf, nb_full_node - 14)
        x_new = t
        # interpolation full data
        # muscles
        f_mus = interp1d(x, result_mat[i]["U_est"][:, :-14])
        full_muscle_est = f_mus(x_new)

        # Q
        f_Q = interp1d(x, result_mat[i]["X_est"][:, :-14])
        full_Q_est = f_Q(x_new)

        f_Q_int = interp1d(x, result_mat[i]["q_int"][:, :-14])
        full_Q_int = f_Q_int(x_new)

        f_kalman = interp1d(x, result_mat[i]["kalman"][:, :-14])
        full_kalman = f_kalman(x_new)

# params = ["full", "mhe", 'ref']
params = ["mhe", "ref"]
params = ["mhe", "ref"]


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

not_interest_muscles = [0, 1, 4, 5, 6, 7, 8, 9, 12, 19, 20, 22]
interest_muscles = []
for i in range(models[0].nbMuscles()):
    if i not in not_interest_muscles:
        interest_muscles += [i]


param_df = ([params[0]] * len(trial) * nb_mhe * len(interest_muscles)
            + [params[1]] * len(trial) * nb_mhe * len(interest_muscles)
            # + [params[2]] * len(trial) * nb_mhe * len(interest_muscles)
             )
full_muscle_est_onset = [0] * models[0].nbMuscles()
mhe_muscle_est_onset = [0] * models[0].nbMuscles()
ref_muscle_est_onset = [0] * models[0].nbMuscles()
full_muscle_est_onset_range = [(0, 0)] * models[0].nbMuscles()
mhe_muscle_est_onset_range = [(0, 0)] * models[0].nbMuscles()
ref_muscle_est_onset_range = [(0, 0)] * models[0].nbMuscles()
for t in trial:
    for param in params:
        for i in interest_muscles:
            if i == interest_muscles[0] and param == params[0] and t == trial[0]:
                muscles_names = ([models[0].muscleNames()[i].to_string()] * nb_mhe)
            else:
                muscles_names += ([models[0].muscleNames()[i].to_string()] * nb_mhe)

count = 0
data_df = np.zeros((len(params) * len(trial) * nb_mhe * len(interest_muscles)))
# names_df = np.ndarray((len(params) * len(trial) * nb_mhe * len(interest_muscles)))
for p in range(len(params)):
    for subj in subject:
        result_dir = f"{subj}/"
        for i in range(len(trial)):
            if params[p] == "full" and "full" in trial[i]:
                full_start = [0] * models[0].nbMuscles()
            elif params[p] == "mhe" and "full" not in trial[i]:
                mhe_start = [0] * models[0].nbMuscles()
            elif params[p] == "ref" and "full" not in trial[i]:
                ref_start = [0] * models[0].nbMuscles()
            for m in interest_muscles:
                file = result_dir + f"{trial[i]}"
                result_mat.append(read_data(file))
                if params[p] == "full" and "full" in trial[i]:
                    for j in range(full_muscle_est.shape[1]):
                        if full_muscle_est[m, j] >= onset_limit:
                            if full_start[m] == 0:
                                full_start[m] = j
                            full_muscle_est_onset[m] += 1
                    data_df[count:count + nb_mhe] = full_muscle_est[m, :]
                    # names_df[count:count + nb_mhe] = np.array([models[0].muscleNames()[m].to_string()]*nb_mhe)

                elif params[p] == "mhe" and "full" not in trial[i]:
                    for j in range(full_muscle_est.shape[1]):
                        if result_mat[i]["U_est"][m, j] >= onset_limit:
                            if mhe_start[m] == 0:
                                mhe_start[m] = j
                            mhe_muscle_est_onset[m] += 1
                    data_df[count:count + nb_mhe] = result_mat[i]["U_est"][m, :]
                    # names_df[count:count + nb_mhe] = np.array([models[0].muscleNames()[m].to_string()]*nb_mhe)
                elif params[p] == "ref" and "full" not in trial[i]:
                    if m in muscle_track_idx:
                        idx = muscle_track_idx.index(m)
                        for j in range(full_muscle_est.shape[1]):
                            if result_mat[i]["muscles_target"][idx, j] >= onset_limit:
                                if ref_start[m] == 0:
                                    ref_start[m] = j
                                ref_muscle_est_onset[m] += 1
                        data_df[count:count + nb_mhe] = result_mat[i]["muscles_target"][idx, :]
                    else:
                        data_df[count:count + nb_mhe] = np.zeros((1, nb_mhe))
                count += nb_mhe
            for m in interest_muscles:
                if params[p] == "full" and "full" in trial[i]:
                    full_muscle_est_onset_range[m] = [(full_start[m], full_muscle_est_onset[m])]
                elif params[p] == "mhe" and "full" not in trial[i]:
                    mhe_muscle_est_onset_range[m] = [(mhe_start[m], mhe_muscle_est_onset[m])]
                elif params[p] == "ref" and "full" not in trial[i]:
                    ref_muscle_est_onset_range[m] = [(ref_start[m], ref_muscle_est_onset[m])]

emg = pd.DataFrame(
    {"params": param_df,
     "data_df": data_df,
     "names": muscles_names,
     # "EMG_mhe" : emg_mhe[0, 0:1],
     # "EMG_ref" : emg_ref[0, 0:1],
     # "names": ["Full", "MHE", "ref"]
})
c=0
emg_hm = np.zeros((len(params) * len(muscle_track_idx), nb_mhe))
for m in range(models[0].nbMuscles()):
    if m in muscle_track_idx:
        idx = muscle_track_idx.index(m)
        emg_hm[c:c+2, :] = np.concatenate((emg_ref[idx, :][np.newaxis, :], emg_mhe[m, :][np.newaxis, :]), axis=0)
        c+=2


c = 0
cp = sns.color_palette("rocket_r", as_cmap=True)
plt.figure()
cbar = [False] * len(muscle_track_idx)
cbar_kws = [None] * len(muscle_track_idx)
cbar[-1] = True
cbar_kws[-1] = {"orientation": "horizontal"}
for i in range(len(muscle_track_idx)):
    plt.subplot(len(muscle_track_idx), 1, i+1)
    ax = sns.heatmap(emg_hm[c:c+2, :], vmin=0, vmax=0.2, cmap=cp)
    # plt.broken_barh(full_muscle_est_onset_range[i], [2 + count, 1])
    # plt.broken_barh(ref_muscle_est_onset_range[i], [3 + count, 1], color='r')
    c += 2


plt.show()
# sns.set_style("whitegrid")
# # cp = sns.color_palette("YlOrRd", 5)
# ax2 = sns.boxplot(y="names", x="data_df", hue="params", data=emg, #orient="h"
#     # palette=cp,
# )
# ax = sns.swarmplot(y="names", x="data_df", hue="params", data=emg,)
# ax2.set(ylabel="Muscles")
# ax2.set(xlabel="Activation level")
# ax2.xaxis.get_label().set_fontsize(10)
# ax2.yaxis.get_label().set_fontsize(10)
# ax2.tick_params(labelsize=10)
# # ax2.legend(title="Activation level for abduction")
# plt.setp(ax2.get_legend().get_texts(), fontsize="22")  # for legend text
# plt.setp(ax2.get_legend().get_title(), fontsize="30")  # for legend title
# plt.title(f"Activation level for abduction", fontsize=20)



# ridge_plot = sns.FacetGrid(emg, row="names", hue="names", aspect=5, height=1.25)
# # Draw the densities in a few steps
# ridge_plot.map(sns.kdeplot, "data_df", clip_on=False, shade=False, alpha=0.7, lw=1, bw_method=.2)
# ridge_plot.map(plt.axhline, y=0, lw=4, clip_on=False)
#
# def label(x, color, label):
#     ax = plt.gca()
#     ax.text(-0.1, .1, label, color="black",
#             ha="left", va="center", transform=ax.transAxes)
#
# ridge_plot.map(label, "data_df")
# # Set the subplots to overlap
# # ridge_plot.fig.subplots_adjust(hspace=-0.01)
# # Remove axes details that don't play well with overlap
# ridge_plot.set_titles("")
# ridge_plot.set(yticks=[])
# #ridge_plot.set_xlabel("CO2 Emission",fontsize=30)
# # ridge_plot.despine(bottom=True, left=True)
# # ridge_plot.savefig("Ridgeline_plot_Seaborn_Python.png")
plt.show()