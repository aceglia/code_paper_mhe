import numpy as np
from biosiglive.data_processing import read_data

try:
    import biorbd
except:
    import biorbd_casadi as biorbd

import os
import matplotlib.pyplot as plt
import seaborn


# --- RMSE --- #
def rmse(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).mean())


def std(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).std())


seaborn.set_style("whitegrid")
seaborn.color_palette()
parent = os.path.dirname(os.getcwd())
subject = ["subject_1"]
trial = ["abd_cocon_result"]
result_mat = []
models = []
for subj in subject:
    result_dir = f"{subj}/"
    for i in range(len(trial)):
        file = result_dir + f"{trial[i]}"
        result_mat.append(read_data(file))
        nb_mhe = result_mat[i]["U_est"].shape[1]
    models.append(biorbd.Model(f"{subj}/model_{subj}_scaled.bioMod"))

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


muscles_names = [
    # "Trapeze Medial",
    # "Trapeze Inferior",
    # "Trapeze Superior",
    # "Subclavicular",
    # "Deltoid Anterior",
    "Deltoid Medial",
    "Supraspinatus",
    "Subscapular",
    "Teres Major",
    "Deltoid Posterior",
    "Biceps Long",
    "Biceps Short",
    "Pectoralis Sternal",
    "Latissimus Dorsi",
    "Triceps Long",
    "Triceps Lateral",
    "Triceps Medial",
]

not_interest_muscle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19, 22, 25]
fontsize = 17
tf = nb_mhe / 35
t = np.linspace(0, tf, nb_mhe)
# PLot muscular force for reference movement and optimal movement(track and minimize EMG)
seaborn.set_style("whitegrid")
seaborn.color_palette()
fig = plt.figure("Muscles activation")
plt.gcf().subplots_adjust(left=0.06, right=0.99, wspace=0.05, hspace=0.25, bottom=0.08, top=0.90)
for s in range(len(trial)):
    count = 0
    c = 0
    for i in range(models[0].nbMuscles()):
        fig = plt.subplot(4, 3, count + 1)
        if i not in not_interest_muscle:
            plt.plot(t, result_mat[s]["U_est"][i, :] * 100, label="Estimated activation")
            if i in muscle_track_idx:
                idx = muscle_track_idx.index(i)
                plt.plot(t, result_mat[s]["muscles_target"][idx, :] * 100, label="Experimental EMG")
            if count in [9, 10, 11]:
                plt.xlabel("Time (s)", fontsize=fontsize)
                fig.set_xlim(0, tf)
            else:
                fig.set_xticklabels([])
            if count in [0, 3, 6, 9]:
                plt.ylabel("Muscle activation \n(% MVC)", fontsize=fontsize - 2)
            else:
                fig.set_yticklabels([])
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.ylim(0, 100)
            plt.gca().set_prop_cycle(None)
            plt.title(muscles_names[count], fontsize=fontsize)
            count += 1

plt.legend(loc="upper center", bbox_to_anchor=(-0.75, 5.3), ncol=2, fontsize=fontsize, frameon=True)
plt.show()
