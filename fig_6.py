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
    conditions = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
    trials = [
        "data_abd_sans_poid",
        "data_abd_poid_2kg",
        "data_flex_poid_2kg",
        "data_cycl_poid_2kg",
        "data_flex_sans_poid",
        "data_cycl_sans_poid",
    ]
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
    rmse_q_full = []
    rmse_q = []
    t_est = []
    t_ref = []
    nb_mhe = []
    n_frames = [0, 25, 50, 75, 100]
    model = biorbd.Model(f"data/wu_scaled.bioMod")
    result_dic_tmp = {}
    result_all_dic = {}
    n_init = [int(0)] * len(conditions)
    with open("results/result_all_trials", "rb") as file:
        while True:
            try:
                data_tmp = pickle.load(file)
                key = list(data_tmp.keys())[0]
                result_all_dic[key] = data_tmp[key]
            except:
                break

    tot_err_mark = np.zeros((len(conditions), len(n_frames)))
    tot_err_emg_phase = np.zeros((len(conditions), len(n_frames)))
    tot_err_emg_mag = np.zeros((len(conditions), len(n_frames)))
    tot_err_ID = np.zeros((len(conditions), len(n_frames)))
    tot_freq = np.zeros((len(conditions), len(n_frames)))
    tot_std_mark = np.zeros((len(conditions), len(n_frames)))
    tot_std_emg_phase = np.zeros((len(conditions), len(n_frames)))
    tot_std_emg_mag = np.zeros((len(conditions), len(n_frames)))
    tot_std_ID = np.zeros((len(conditions), len(n_frames)))
    tot_std_freq = np.zeros((len(conditions), len(n_frames)))
    for c, cond in enumerate(conditions):
        for f, frame in enumerate(n_frames):
            err_id_tmp, err_tmp_mark, err_tmp_emg_mag, err_tmp_emg_phase, freq_tmp = [], [], [], [], []
            for key in result_all_dic.keys():
                if np.isfinite(result_all_dic[key][str(cond)][str(frame)]["rmse_torque"]):
                    err_id_tmp.append(result_all_dic[key][str(cond)][str(frame)]["rmse_torque"])
                    err_tmp_emg_mag.append(result_all_dic[key][str(cond)][str(frame)]["magnitude_emg_err"])
                    err_tmp_emg_phase.append(result_all_dic[key][str(cond)][str(frame)]["phase_emg_err"])
                    err_tmp_mark.append(result_all_dic[key][str(cond)][str(frame)]["rmse_markers"])
                    freq_tmp.append(result_all_dic[key][str(cond)][str(frame)]["sol_freq_mean"])
                else:
                    print(key, cond, frame, result_all_dic[key][str(cond)][str(frame)]["rmse_torque"])
            tot_err_ID[c, f] = np.mean(err_id_tmp)
            tot_err_mark[c, f] = np.mean(err_tmp_mark)
            tot_err_emg_phase[c, f] = np.mean(err_tmp_emg_phase)
            tot_err_emg_mag[c, f] = np.mean(err_tmp_emg_mag)
            tot_freq[c, f] = np.mean(freq_tmp)
            tot_std_ID[c, f] = np.std(err_id_tmp)
            tot_std_mark[c, f] = np.std(err_tmp_mark)
            tot_std_emg_phase[c, f] = np.std(err_tmp_emg_phase)
            tot_std_emg_mag[c, f] = np.std(err_tmp_emg_mag)
            tot_std_freq[c, f] = np.std(freq_tmp)
            print(f"error_markers: {tot_err_mark[c, f]} +/- {tot_std_mark[c, f]}")
            print(f"error_torque: {tot_err_ID[c, f]} +/- {tot_std_ID[c, f]}")
            print(f"error_emg: {tot_err_emg_phase[c, f]} +/- {tot_std_emg_phase[c, f]}")

    # import matplotlib.pyplot as plt
    # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    #
    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    #
    #
    # def getImage(path):
    #     return OffsetImage(plt.imread(path, format="png"), zoom=.02
    #                     )
    #
    #
    # paths = ['figure_img/0_10.png', 'figure_img/0_10.png', 'figure_img/0_10.png', 'figure_img/0_10.png']
    # x = [8, 4, 3, 6]
    # y = [5, 3, 4, 7]
    # fig, ax = plt.subplots()
    # for x0, y0, path in zip(x, y, paths):
    #     ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    #     ax.add_artist(ab)
    # plt.xticks(range(10))
    # plt.yticks(range(10))
    # plt.show()


    # fig = plt.figure(num="fig_errors", constrained_layout=True)
    fig, axs = plt.subplots(2, 2)
    fig_names = ["a) Markers", "b) EMG", "c) Inverse dynamics"]
    y_labels = ["RMSE (m)", "Phase error (%)", "RMSE (N.m)"]
    x_label = "Time to solve a subproblem (ms)"
    errors = [tot_err_mark, tot_err_emg_phase, tot_err_ID]
    color = seaborn.color_palette()
    axs.resize(4, 1)
    fig.delaxes(axs[-1][0])

    for i in range(3):
        ax = axs[i][0]
        ax.set_title(fig_names[i])
        for c, cond in enumerate(conditions):
            for f, frame in enumerate(n_frames):
                ax.scatter(1 / tot_freq[c, f] * 1000, errors[i][c, f], color=color[c], alpha=0.2 + 0.1 * f)
                ax.set_ylabel(y_labels[i])
                ax.set_xlabel(x_label)
        # plt.figure("emg magnitude error")
        # for c, cond in enumerate(conditions):
        #     for f, frame in enumerate(n_frames):
        #         if f == len(n_frames) - 1:
        #             plt.scatter(
        #                 1 / tot_freq[c, f] * 1000,
        #                 tot_err_emg_mag[c, f],
        #                 color=color[c],
        #                 alpha=0.2 + 0.1 * f,
        #                 label=f"{cond}",
        #             )
        #         else:
        #             plt.scatter(1 / tot_freq[c, f] * 1000, tot_err_emg_mag[c, f], color=color[c], alpha=0.2 + 0.1 * f)
    plt.show()
