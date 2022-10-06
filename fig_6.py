import numpy as np
import seaborn
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import biorbd
import matplotlib.pyplot as plt


# --- RMSE --- #
def rmse(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).mean())


def std(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).std())


if __name__ == "__main__":
    conditions = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
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
    with open("results/results_all_trials_w6_freq", "rb") as file:
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
    tot_delay = np.zeros((len(conditions), len(n_frames)))
    tot_saturation  = np.zeros((len(conditions), len(n_frames)))
    tot_grad  = np.zeros((len(conditions), len(n_frames)))
    for c, cond in enumerate(conditions):
        for f, frame in enumerate(n_frames):
            err_id_tmp, err_tmp_mark, err_tmp_emg_mag, err_tmp_emg_phase, freq_tmp, saturation_tmp, grad_tmp  = [], [], [], [], [], [], []
            for key in result_all_dic.keys():
                if np.isfinite(result_all_dic[key][str(cond)][str(frame)]["rmse_torque"]):
                    err_id_tmp.append(result_all_dic[key][str(cond)][str(frame)]["rmse_torque"])
                    err_tmp_emg_mag.append(result_all_dic[key][str(cond)][str(frame)]["magnitude_emg_err"])
                    err_tmp_emg_phase.append(result_all_dic[key][str(cond)][str(frame)]["phase_emg_err"])
                    err_tmp_mark.append(result_all_dic[key][str(cond)][str(frame)]["rmse_markers"])
                    freq_tmp.append(result_all_dic[key][str(cond)][str(frame)]["sol_freq_mean"])
                    saturation_tmp.append(result_all_dic[key][str(cond)][str(frame)]["saturation"])
                    grad_tmp.append(result_all_dic[key][str(cond)][str(frame)]["gradient"])
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
            tot_delay[c, f] = (1 / tot_freq[c, f] + cond * (1 - (0.01 * frame))) * 1000
            tot_saturation[c, f] = np.mean(saturation_tmp)
            tot_grad[c, f] = np.mean(grad_tmp)

            print(f"error_markers: {tot_err_mark[c, f]} +/- {tot_std_mark[c, f]}")
            print(f"error_torque: {tot_err_ID[c, f]} +/- {tot_std_ID[c, f]}")
            print(f"error_emg: {tot_err_emg_phase[c, f]} +/- {tot_std_emg_phase[c, f]}")
    from PIL import ImageColor
    from matplotlib.ticker import FormatStrFormatter
    def getImage(path, alpha=1, scale=0.03, color_wanted=(0,0,0)):
        from matplotlib import cbook, colors as mcolors

        from matplotlib.image import AxesImage
        import matplotlib.pyplot as plt
        from matplotlib.transforms import Bbox, TransformedBbox, BboxTransformTo
        original_image = plt.imread(path)
        cut_location = 70
        b_and_h = original_image[:, :, 2:3]
        color = original_image[:, :, 2:3] - original_image[:, :, 0:1]
        color = np.ones((color.shape[0], color.shape[1], color.shape[2]))*0.4
        alpha = original_image[:, :, 3:4]
        nx = original_image.shape[1]
        rgb = mcolors.to_rgba(color_wanted)[:3]
        im = np.dstack(
            #[b_and_h + color, alpha])
            [b_and_h + color * (1-np.array(rgb)), alpha])
        return OffsetImage(im, zoom=scale)
        # return OffsetImage(plt.imread(path, format="png"), zoom=scale,alpha=0.4 + 0.1 * alpha)
    colors = seaborn.color_palette()
    fig, axs = plt.subplots(3, 2)
    fig_names = ["a) Markers", "b) EMG", "c) Inverse dynamics", "d) Total delay", "e) Saturation", "f) Noise"]
    y_labels = ["RMSE (mm)", "Phase error (%)", "RMSE (N.m)", "Delay (ms)", "Saturation (%)", "Gradient"]
    x_label = "Time to solve a subproblem (ms)"
    errors = [tot_err_mark * 1000, tot_err_emg_phase, tot_err_ID, tot_delay, tot_saturation, tot_grad]
    img_path = "results/img/frame_"
    color = "white"
    axs.resize(6, 1)
    delay = 60
    duration = 1 / tot_freq * 1000
    delta = 1.5
    c, f = 0, 0
    x = np.linspace(duration.min()-delta, duration.max()+delta, 100)
    line_delta = [-1.5, -0.75, 0, 0.75, 1.5]
    x_ticks = []
    for i in range(6):
        ax = axs[i][0]
        ax.set_title(fig_names[i])
        for c, cond in enumerate(conditions):
            for f, frame in enumerate(n_frames):
                if i == 3:
                    ax.plot(x, [delay] * 100, alpha=0.5, linewidth=0.5)
                    x_ticks.append(1 / tot_freq[c, f] * 1000)
                ax.axvline(x=1 / tot_freq[c, f] * 1000, linestyle='dotted', color='grey', linewidth=0.5 )
                ax.scatter(1 / tot_freq[c, f] * 1000 + line_delta[f], errors[i][c, f], color=color, alpha=0.2 + 0.1 * f)
                ab = AnnotationBbox(getImage(img_path + f"{frame}.png", color_wanted=colors[c]), (1 / tot_freq[c, f] * 1000 + line_delta[f], errors[i][c, f]),
                                    frameon=False)
                ax.add_artist(ab)

        ax.set_ylabel(y_labels[i])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if i in [4, 5]:
            ax.set_xlabel(x_label)
            ax.set_xticks(x_ticks)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

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
