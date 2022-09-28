import numpy as np
import seaborn
import pickle

import biorbd
import matplotlib.pyplot as plt


# --- RMSE --- #
def rmse(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).mean())


def std(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).std())


if __name__ == "__main__":
    trials = ["data_abd_sans_poid", "data_abd_poid_2kg"]
    result_mat = []

    muscle_track_idx = [
        14,
        23,
        24,  # MVC Pectoralis sternalis
        13,  # MVC Deltoid anterior
        15,  # MVC Deltoid medial
        16,  # MVC Deltoid posterior
        26,
        27,  # MVC Biceps brachii
        28,
        29,
        30,  # MVC Triceps brachii
        11,  # MVC Trapezius superior
        1,  # MVC Trapezius superior bis
        2,  # MVC Trapezius medial
        3,  # MVC Trapezius inferior
        25,  # MVC Latissimus dorsi
    ]

    interest_muscle = [11, 13, 15, 16, 17, 18, 19, 23]
    model = biorbd.Model(f"data/wu_scaled.bioMod")
    result_dic_tmp = {}
    result_all_dic = {}
    with open("results/result_abd_75", "rb") as file:
        while True:
            try:
                data_tmp = pickle.load(file)
                key = list(data_tmp.keys())[0]
                result_all_dic[key] = data_tmp[key]
            except:
                break

    muscle_names = [
        "Trapeze superior",
        "Deltoid anterior",
        "Deltoid medial",
        "Deltoid posterior",
        "Supraspinatus",
        "Infraspinatus",
        "Subscapularis",
        "Pectoral medial",
    ]

    bw = 720
    color = seaborn.color_palette()
    fig = plt.figure(num="fig_muscle", constrained_layout=True)
    subfigs = fig.subfigures(4, 3, wspace=0.05, hspace=0.1)
    rmse_emg = []
    rmse_emg_full = []
    labels = ["Forces without weight", "Forces with 2 kg weight"]
    labels_target = ["Recorded EMG without weight", "Recorded EMG with 2 kg weight"]
    labels_est = ["Estimated EMG without weight", "Estimated EMG with 2 kg weight"]
    n_split = [[50, 290], [50, 245]]
    g = 0
    # plt.figure("Estimated activation and EMG signals" + key)
    cond = 0.09
    frame = 75
    count = 0
    b = 0
    for i in range(model.nbMuscles()):
        if i in interest_muscle:
            axs = subfigs.flat[b].subplots(2, 1)
            b += 1
            for k, key in enumerate(result_all_dic.keys()):
                t = result_all_dic[key][str(cond)][str(frame)]["t_est"][n_split[k][0] : n_split[k][1]]
                t = np.linspace(0, 100, t.shape[0])
                if i in muscle_track_idx:
                    idx = muscle_track_idx.index(i)
                    axs.flat[0].plot(
                        t,
                        result_all_dic[key][str(cond)][str(frame)]["muscles_target"][
                            idx, n_split[k][0] : n_split[k][1]
                        ],
                        "--",
                        label=labels_target[k],
                        color=color[k],
                    )
                axs.flat[0].set_title(muscle_names[b - 1], fontsize=12)
                axs.flat[0].plot(
                    t,
                    result_all_dic[key][str(cond)][str(frame)]["U_est"][i, n_split[k][0] : n_split[k][1]],
                    label=labels_est[k],
                    alpha=0.8,
                    color=color[k],
                )
                axs.flat[1].plot(
                    t,
                    result_all_dic[key][str(cond)][str(frame)]["f_est"][i, n_split[k][0] : n_split[k][1]] / bw,
                    label=labels[k],
                    color=color[k],
                )
                axs.flat[0].set_xticklabels([])
                axs.flat[1].set_xticklabels([])
                if b - 1 in [0, 3, 6]:
                    axs.flat[0].set_ylabel("Activations\n (% MVC)\n", fontsize=12)
                    axs.flat[1].set_ylabel("Force\n (% BW)\n", fontsize=12)
                else:
                    axs.flat[0].set_yticklabels([])
                    axs.flat[1].set_yticklabels([])
                axs.flat[0].set_ylim(0, 1)
                axs.flat[1].set_ylim(0, 1)
                axs.flat[0].grid(True)
                axs.flat[1].grid(True)
                count += 1
    for j in range(3):
        if g == 0:
            axb = subfigs.flat[b].subplots(1, 1)
            b += 1
            t1 = result_all_dic[list(result_all_dic.keys())[0]][str(cond)][str(frame)]["t_est"][
                n_split[0][0] : n_split[0][1]
            ]
            t1 = np.linspace(0, 100, t1.shape[0])
            t2 = result_all_dic[list(result_all_dic.keys())[1]][str(cond)][str(frame)]["t_est"][
                n_split[1][0] : n_split[1][1]
            ]
            t2 = np.linspace(0, 100, t2.shape[0])
            axb.plot(
                t1,
                result_all_dic[list(result_all_dic.keys())[0]][str(cond)][str(frame)]["X_est"][
                    6, n_split[0][0] : n_split[0][1]
                ]
                * (180 / np.pi),
                label=cond,
                color=color[0],
            )
            axb.plot(
                t2,
                result_all_dic[list(result_all_dic.keys())[1]][str(cond)][str(frame)]["X_est"][
                    6, n_split[1][0] : n_split[1][1]
                ]
                * (180 / np.pi),
                color=color[1],
                label=cond,
            )

            axb.set_xlabel("Shoulder abduction (%)", fontsize=12)
            if j == 1:
                axb.set_ylabel("Joint angle (°)\n", fontsize=12)
            else:
                axb.set_yticklabels([])
            axb.set_ylim(0, 100)
            axb.grid(True)
    plt.show()
