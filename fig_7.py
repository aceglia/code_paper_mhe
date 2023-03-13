import numpy as np
import seaborn
import pickle

import biorbd
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import glob
import opensim as osim


# --- RMSE --- #
def rmse(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).mean())


def std(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).std())

def moment_arm_osim(model_path, q):
    #5-11-12
    q_tmp = q[2:, :]
    q = np.concatenate((q[:2, :], np.zeros((1, q.shape[1]))))
    q = np.concatenate((q, q_tmp))
    q = np.concatenate((q, np.zeros((2, q.shape[1]))))
    q = q #* (180/np.pi)
    model = osim.Model(model_path)
    state = model.initSystem()
    dof_names = []
    muscle_names = []
    # compute moment arm for each muscle
    moment_arm = np.zeros((model.getCoordinateSet().getSize(), model.getMuscles().getSize(), q.shape[1]))
    for j in range(6, model.getCoordinateSet().getSize()):
        if j in [2+6, 16+6, 18+6]:
            pass
        else:
            for k in range(model.getMuscles().getSize()):
                for i in range(q.shape[1]):
                    state.setQ(osim.Vector(q[:, i]))
                    moment_arm[j, k, i] = model.getMuscles().get(k).computeMomentArm(state, model.getCoordinateSet().get(j))
                muscle_names.append(model.getMuscles().get(k).toString())
            dof_names.append(model.getCoordinateSet().get(j).toString())
    return moment_arm, dof_names, muscle_names


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

    interest_muscle = [11, 23, 13, 15, 16, 17, 18, 19]
    model = biorbd.Model(f"data/wu_scaled.bioMod")
    result_dic_tmp = {}
    result_all_dic = {}
    with open("results/results_abd_w9_75_008", "rb") as file:
        while True:
            try:
                data_tmp = pickle.load(file)
                key = list(data_tmp.keys())[0]
                if key in trials:
                    result_all_dic[key] = data_tmp[key]
            except:
                break

    muscles = np.ndarray((3 * 3, 100))
    muscles_wickham_tmp = []
    muscles_wickham = [0]*6
    x = np.linspace(0, 100, 100)
    for file in glob.glob("data/from_wickham/**.csv"):
        data_tmp = pd.read_csv(file)
        idx = np.where(data_tmp.values == 1000000)
        muscles_wickham_tmp.append(data_tmp.values[:idx[0][0], :])
        muscles_wickham_tmp.append(data_tmp.values[idx[0][0] + 1:, :])

        # muscles_wickham_tmp.append(pd.read_csv(file))
        # idx = np.where(muscles_wickham_tmp[-1].values == 1000000)
        # muscles_wickham_tmp[-1].values[idx[0][0], :] = np.nan

    muscles_wickham[0] = muscles_wickham_tmp[2]
    muscles_wickham[1] = muscles_wickham_tmp[3]
    muscles_wickham[2] = muscles_wickham_tmp[0]
    muscles_wickham[3] = muscles_wickham_tmp[1]
    muscles_wickham[4] = muscles_wickham_tmp[4]
    muscles_wickham[5] = muscles_wickham_tmp[5]

    muscle_names = [
        "Trapezius (superior)",
        "Pectoralis (medial)",
        "Deltoid (anterior)",
        "Deltoid (medial)",
        "Deltoid (posterior)",
        "Supraspinatus",
        "Infraspinatus",
        "Subscapularis",

    ]
    fontsize = 12
    bw = 720
    color = seaborn.color_palette()
    fig = plt.figure(num="fig_muscle", constrained_layout=True)
    subfigs = fig.subfigures(4, 3, width_ratios=[1, 1, 1], wspace=.02)
    rmse_emg = []
    rmse_emg_full = []
    labels = ["Forces without weight", "Forces with 2 kg weight"]
    labels_target = ["Recorded EMG without weight", "Recorded EMG with 2 kg weight"]
    labels_est = ["Estimated EMG without weight", "Estimated EMG with 2 kg weight"]
    n_split = [[50, 277+50], [50, 227+50]]
    g = 0
    # plt.figure("Estimated activation and EMG signals" + key)
    cond = 0.08
    frame = 75
    count = 0
    b = 0
    linestyles = OrderedDict([('loosely dashed', (0, (2, 2)))])
    for i in interest_muscle:
        axs = subfigs.flat[b].subplots(2, 1)
        b += 1
        for k, key in enumerate(result_all_dic.keys()):
            t = result_all_dic[key][str(cond)][str(frame)]["t_est"][n_split[k][0] : n_split[k][1]]
            t = np.linspace(0, 100, t.shape[0])

            axs.flat[0].set_title(muscle_names[b - 1], fontsize=fontsize)
            # axs.flat[0].set_title(model.muscleNames()[i].to_string(), fontsize=fontsize)
            axs.flat[0].plot(
                t,
                result_all_dic[key][str(cond)][str(frame)]["U_est"][i, n_split[k][0] : n_split[k][1]] * 100,
                label=labels_est[k],
                #alpha=0.8,
                color=color[k],

            )
            if b in [6, 7, 8]:
                if k == 0:
                    print(count)
                    axs.flat[0].plot(
                        np.linspace(0, 50, muscles_wickham[count].shape[0]),
                        muscles_wickham[count][:, 2],
                        linestyle=linestyles["loosely dashed"],
                        label=labels_est[k],
                        alpha=0.4,
                        color="r",
                        linewidth=3
                    )
                    axs.flat[0].fill_between(
                        np.linspace(0, 50, muscles_wickham[count].shape[0]),
                        muscles_wickham[count][:, 1],
                        muscles_wickham[count][:, 3],
                        label=labels_est[k],
                        alpha=0.2,
                        color="r",
                    )
                    axs.flat[0].plot(
                        np.linspace(50, 100, muscles_wickham[count + 1].shape[0]),
                        muscles_wickham[count + 1][:, 2],
                        linestyle=linestyles["loosely dashed"],
                        label=labels_est[k],
                        alpha=0.4,
                        color="r",
                        linewidth=3
                    )
                    axs.flat[0].fill_between(
                        np.linspace(50, 100, muscles_wickham[count + 1].shape[0]),
                        muscles_wickham[count + 1][:, 1],
                        muscles_wickham[count + 1][:, 3],
                        label=labels_est[k],
                        alpha=0.2,
                        color="r",
                    )
                    count += 2
            axs.flat[1].plot(
                t,
                result_all_dic[key][str(cond)][str(frame)]["f_est"][i, n_split[k][0] : n_split[k][1]] / bw * 100,
                label=labels[k],
                color=color[k],
            )
            if i in muscle_track_idx:
                idx = muscle_track_idx.index(i)
                axs.flat[0].plot(
                    t,
                    result_all_dic[key][str(cond)][str(frame)]["muscles_target"][
                    idx, n_split[k][0]: n_split[k][1]
                    ] * 100,
                    linestyle=linestyles["loosely dashed"],
                    label=labels_target[k],
                    color=color[k],
                    linewidth=3
                )
            yticks = np.linspace(0, 1, 3)
            axs.flat[0].set_xticklabels([])
            axs.flat[1].set_xticklabels([])
            if b - 1 in [0, 3, 6]:
                # axs.flat[0].set_ylabel("", fontsize=fontsize+5, rotation=90)
                # axs.flat[1].set_ylabel("", fontsize=fontsize+5, rotation=90)
                axs.flat[0].tick_params(axis='y', labelsize=fontsize-2)
                axs.flat[1].tick_params(axis='y', labelsize=fontsize-2)
            else:
                axs.flat[0].tick_params(axis='y', labelsize=fontsize-2, colors="w")
                axs.flat[1].tick_params(axis='y', labelsize=fontsize-2, colors="w")
                # axs.flat[0].set_yticks(yticks)
                # axs.flat[1].set_yticks(yticks)
                # axs.flat[1].set_yticklabels([])
                # axs.flat[0].set_yticklabels([])
            axs.flat[0].set_ylim(0, 100)
            axs.flat[1].set_ylim(0, 100)
            # axs.flat[0].set_xlim(0, 100)
            # axs.flat[1].set_xlim(0, 100)
            axs.flat[0].grid(True)
            axs.flat[1].grid(True)

    for j in range(3):
        if g == 0:
            axb = subfigs.flat[b].subplots(1, 1)
            axb.set_title("Arm elevation", fontsize=fontsize)
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
                t1,
                result_all_dic[list(result_all_dic.keys())[0]][str(cond)][str(frame)]["kalman"][
                    6, n_split[0][0] : n_split[0][1]
                ]
                * (180 / np.pi),
                label=cond,
                color=color[0],
                linestyle="dashdot",
                linewidth=2,

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
            axb.plot(
                t2,
                result_all_dic[list(result_all_dic.keys())[1]][str(cond)][str(frame)]["kalman"][
                    6, n_split[1][0] : n_split[1][1]
                ]
                * (180 / np.pi),
                color=color[1],
                label=cond,
                linestyle="dashdot",
                linewidth=2
            )
            yticks = np.linspace(0, 100, 6)
            axb.set_xlabel("Soulder abduction (%)", fontsize=fontsize)
            axb.tick_params(axis='x', labelsize=fontsize - 2)
            if j == 1:
                # axb.set_ylabel("", fontsize=fontsize, rotation=0)
                axb.tick_params(axis='y', labelsize=fontsize-2)
            else:
                axb.tick_params(axis='y', labelsize=fontsize - 2, colors="w")
            axb.set_ylim(0, 100)
            # axb.set_xlim(0, 100)
            axb.grid(True)
    # plt.figure("All muscles")
    # k = 0
    # key = list(result_all_dic.keys())[0]
    # q = result_all_dic[key][str(cond)][str(frame)]["X_est"][:10, n_split[k][0]: n_split[k][1]]
    # osim_moment_arm, dof_names, muscle_names = moment_arm_osim(
    #     f"/home/amedeoceglia/Documents/programmation/code_paper_mhe_data/data_final_new/subject_3/wu_scaled.osim", q)
    # osim_q_idx = []
    # for n, name in enumerate(model.nameDof()):
    #     for o, osim_name in enumerate(dof_names):
    #         if osim_name in name.to_string():states
    #             osim_q_idx.append(o)
    # for q in range(model.nbQ()):
    #     plt.figure(f"{model.nameDof()[q].to_string()}_{dof_names[q]}")
    #
    #     for i in range(model.nbMuscles()):
    #         plt.subplot(7, 5, i+1)
    #         t = result_all_dic[key][str(cond)][str(frame)]["t_est"][n_split[k][0]: n_split[k][1]]
    #         t = np.linspace(0, 100, t.shape[0])
    #
    #         osim_idx = muscle_names.index(model.muscleNames()[i].to_string())
    #         plt.plot(
    #             t,
    #             osim_moment_arm[q, osim_idx, : ] * 100,
    #             label=labels_est[k],
    #             alpha=0.8,
    #             color="r",
    #         )
    #         # axs.flat[0].set_title(model.muscleNames()[i].to_string() , fontsize=fontsize)
    #
    #         plt.plot(
    #             t,
    #             result_all_dic[key][str(cond)][str(frame)]["muscle_moment_arm"][i, q, n_split[k][0]: n_split[k][1]] * 100,
    #             label=labels_est[k],
    #             alpha=0.8,
    #             color=color[k],
    #
    #         )
    #         plt.title(model.muscleNames()[i].to_string() + "_" + muscle_names[osim_idx], fontsize=fontsize)
    # define the size of the plot
    fig.set_size_inches(9.5, 12)
    plt.show()
