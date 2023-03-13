import numpy as np
import seaborn
import pickle
import biorbd
import matplotlib.pyplot as plt
from collections import OrderedDict


if __name__ == "__main__":
    trials = ["data_abd_sans_poid"]
    result_mat = []
    model = biorbd.Model(f"data/wu_scaled.bioMod")
    result_dic_tmp = {}
    result_all_dic = {}
    with open("results/results_all_trials_w6_freq_calc.bio", "rb") as file:
        while True:
            try:
                data_tmp = pickle.load(file)
                key = list(data_tmp.keys())[0]
                result_all_dic[key] = data_tmp[key]
            except:
                break

    fontsize = 16
    color = seaborn.color_palette()
    color.pop(3)
    fig = plt.figure(num="fig_kin")
    labels = ["Forces without weight", "Forces with 2 kg weight"]
    labels_target = ["Recorded EMG without weight", "Recorded EMG with 2 kg weight"]
    labels_est = ["Estimated EMG without weight", "Estimated EMG with 2 kg weight"]
    n_split = [[50, 277+50], [50, 227+50]]
    g = 0
    # plt.figure("Estimated activation and EMG signals" + key)
    cond = [0.04, 0.05, 0.06, 0.07, 0.08]
    frame = 75
    count = 0
    b = 0
    idx = [6, 16]
    titles = ["Arm elevation", "Joint angular velocity"]
    y_labels = ["Abduction angle (°)", "Joint angular velocity (°/s)"]
    linestyles = OrderedDict([('loosely dashed', (0, (2, 2)))])
    alpha = [0.4, 0.5, 0.6, 0.7, 1, 1]
    plt.grid(visible=True)
    for i in range(2):
        x = np.linspace(0, 150, result_all_dic["data_abd_sans_poid"][str(0.04)][str(frame)]["kalman"].shape[1])
        plt.plot(x, result_all_dic["data_abd_sans_poid"][str(0.04)][str(frame)]["kalman"][idx[i], :] * (180 / np.pi),
                 label="kalman",
                 color='r',
                 linestyle='dashed')
        for c, con in enumerate(cond):
            plt.subplot(2, 1, i+1)
            x = np.linspace(0, 150, result_all_dic["data_abd_sans_poid"][str(con)][str(frame)]["X_est"].shape[1])
            linewidth = None if cond != 0.08 else 2
            plt.plot(x, result_all_dic["data_abd_sans_poid"][str(con)][str(frame)]["X_est"][idx[i], :] * (180 / np.pi), label="t_mhe " + str(int(con*1000)),
                     color=color[c],
                     alpha=alpha[c],
                     linewidth=2
                     )
        plt.xlim(0, 100)
        plt.ylabel(y_labels[i], fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(titles[i], fontsize=fontsize)
        if i == 0:
            plt.legend(fontsize=fontsize)
            plt.xticks([])
        else:
            plt.xlabel("Abduction movement (%)", fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
    plt.show()