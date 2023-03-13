import numpy as np
import seaborn
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import biorbd
import matplotlib.pyplot as plt
import codecs
from colormap import rgb2hex


# --- RMSE --- #
def rmse(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).mean())


def std(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).std())


def change_svg(c, condition, f, colors, i, grey=False, opacity=1):
    if grey == 0:
        grey = False
    path = "results/img/frame_"
    old_file = path + f"{f}.svg"
    to_replace_opacity = "opacity:1"
    with codecs.open(old_file, encoding='utf-8', errors='ignore') as fle:
        content = fle.read()
    if grey == 0:
        to_replace = "fill:#989898"
        new_file = content.replace(to_replace,
                                   f"fill:{rgb2hex(int(255 * colors[c][0]), int(255 * colors[c][1]), int(255 * colors[c][2]))}")
    else:
        new_file = content
    new_file = new_file.replace(to_replace_opacity, f"opacity:{opacity}")
    with open(path + f"{f}_{str(condition).replace('.', '')}_{i}.svg", 'w') as f:
        f.write(new_file)
    return path + f"{f}_{str(condition).replace('.', '')}_{i}.svg"


if __name__ == "__main__":
    conditions = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
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
    with open("results/results_all_trials_w6_freq_calc.bio", "rb") as file:
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
    tot_saturation = np.zeros((len(conditions), len(n_frames)))
    tot_grad = np.zeros((len(conditions), len(n_frames)))
    tot_mean_freq = np.zeros((len(conditions), len(n_frames)))
    tot_mean_freq_ref = np.zeros((len(conditions), len(n_frames)))
    for c, cond in enumerate(conditions):
        for f, frame in enumerate(n_frames):
            err_id_tmp, err_tmp_mark, err_tmp_emg_mag, err_tmp_emg_phase, freq_tmp, saturation_tmp, grad_tmp, mean_freq_tmp, mean_freq_tmp_ref = [], [], [], [], [], [], [], [], []
            for key in result_all_dic.keys():
                if np.isfinite(result_all_dic[key][str(cond)][str(frame)]["rmse_torque"]):
                    err_id_tmp.append(result_all_dic[key][str(cond)][str(frame)]["rmse_torque"])
                    err_tmp_emg_mag.append(result_all_dic[key][str(cond)][str(frame)]["magnitude_emg_err"])
                    err_tmp_emg_phase.append(result_all_dic[key][str(cond)][str(frame)]["rmse_emg"])
                    err_tmp_mark.append(result_all_dic[key][str(cond)][str(frame)]["rmse_markers"])
                    freq_tmp.append(result_all_dic[key][str(cond)][str(frame)]["sol_freq_mean"])
                    saturation_tmp.append(result_all_dic[key][str(cond)][str(frame)]["saturation"])
                    mean_freq_tmp.append(result_all_dic[key][str(cond)][str(frame)]["mean_freq_qdot"])
            tot_err_ID[c, f] = np.mean(err_id_tmp)
            tot_err_mark[c, f] = np.mean(err_tmp_mark)
            tot_err_emg_phase[c, f] = np.mean(err_tmp_emg_phase)
            tot_err_emg_mag[c, f] = np.mean(err_tmp_emg_mag)
            tot_freq[c, f] = np.mean(freq_tmp)
            tot_freq[c, f] = np.mean(freq_tmp)
            tot_std_ID[c, f] = np.std(err_id_tmp)
            tot_std_mark[c, f] = np.std(err_tmp_mark)
            tot_std_emg_phase[c, f] = np.std(err_tmp_emg_phase)
            tot_std_emg_mag[c, f] = np.std(err_tmp_emg_mag)
            tot_std_freq[c, f] = np.std(freq_tmp)
            tot_delay[c, f] = (1 / tot_freq[c, f] + cond * (1 - (0.01 * frame))) * 1000
            tot_saturation[c, f] = np.mean(saturation_tmp)
            tot_grad[c, f] = np.mean(grad_tmp)
            tot_mean_freq[c, f] = np.mean(mean_freq_tmp)

            print(f"error_markers: {tot_err_mark[c, f]} +/- {tot_std_mark[c, f]}")
            print(f"error_torque: {tot_err_ID[c, f]} +/- {tot_std_ID[c, f]}")
            print(f"error_emg: {tot_err_emg_phase[c, f]} +/- {tot_std_emg_phase[c, f]}")

    from matplotlib.ticker import FormatStrFormatter
    import skunk
    def getImage(path, alpha=1, scale=1, color_wanted=(0,0,0),use_skunk=True):
        if use_skunk:
            import skunk
            return skunk.Box(50, 50, path[:-4])
        else:
            from matplotlib import cbook, colors as mcolors

            import matplotlib.pyplot as plt
            original_image = plt.imread(path)
            b_and_h = original_image[:, :, 2:3]
            color = original_image[:, :, 2:3] - original_image[:, :, 0:1]
            color = np.ones((color.shape[0], color.shape[1], color.shape[2]))*0.4
            alpha = original_image[:, :, 3:4]
            nx = original_image.shape[1]
            rgb = mcolors.to_rgba(color_wanted)[:3]
            im = np.dstack(
                [b_and_h + color * (1-np.array(rgb)), alpha])
            return OffsetImage(im, zoom=scale)
    colors = seaborn.color_palette()
    fig, axs = plt.subplots(2, 3)
    plt.subplots_adjust(hspace=0)

    fig_names = ["a) Optimisation delay", "d) Markers position error",
                 "b) Muscle activation saturation", "e) Muscle activation error",
                 "c) Joint velocity mean power frequency", "f) Joint torque difference"]
    y_labels = ["Delay (ms)", "RMSE (mm)", "Saturation (%)", "RMSE (%)", "Mean power frequency (Hz)",  "RMSD (N.m)"]
    x_label = "Time to solve a subproblem (ms)"
    errors = [tot_delay, tot_err_mark * 1000, tot_saturation, tot_err_emg_phase * 100, tot_mean_freq, tot_err_ID]
    img_path = "results/img/frame_"
    color = "white"
    axs.resize(6, 1)
    duration = 1 / tot_freq * 1000
    delta = 1.5
    c, f = 0, 0
    x = np.linspace(duration.min()-delta, duration.max()+delta, 100)
    line_delta = [-1.6, -0.85, 0, 0.85, 1.6]
    line_delta = [d for d in line_delta]
    line_delta = [0] * 5
    chosen_point = [0.08, 75]
    fontsize = 12
    x_ticks = []
    skunk_dic = {}
    cmap = plt.get_cmap("rainbow")
    colors = cmap(np.linspace(0, 1, 10))
    import skunk
    grey_mat = np.zeros((7, len(conditions), len(n_frames)))
    for i in range(6):
        ax = axs[i][0]
        ax.set_title(fig_names[i], fontsize=fontsize)
        for c, cond in enumerate(conditions):
            for f, frame in enumerate(n_frames):
                lim_inf = 50
                opacity = 1
                if i == 0:
                    ax.axhspan(ymin=60, ymax=175, alpha=0.005, color="red")
                    ax.set_ylim(15, 175)
                    lim_inf = 58

                    if errors[i][c, f] > lim_inf:
                        grey_mat[i + 2, c, f] = True
                elif i == 2:
                    ax.axhspan(ymin=4, ymax=7, alpha=0.005, color="red")
                    ax.set_ylim(0, 7)
                    lim_inf = 4
                    grey_mat[i + 2, c, f] = grey_mat[i, c, f]
                    if errors[i][c, f] > lim_inf:
                        grey_mat[i + 2, c, f] = True
                if grey_mat[i, c, f] == 1:
                    opacity = 0.5
                svg_path = change_svg(c, cond, frame, colors, i, grey=grey_mat[i, c, f], opacity=opacity)
                ax.axvline(x=1 / tot_freq[c, f] * 1000, color=[i for i in colors[c]], linewidth=0.9, alpha=0.2)
                ax.scatter(1 / tot_freq[c, f] * 1000 + line_delta[f], errors[i][c, f], color=color, alpha=0, s=600)
                ab = AnnotationBbox(skunk.Box(25, 25, f"{i}_{c}_{f}"), (1 / tot_freq[c, f] * 1000 + line_delta[f], errors[i][c, f]),
                                    frameon=False, alpha=0.1)

                ax.add_artist(ab)
                skunk_dic[f"{i}_{c}_{f}"] = f"/home/amedeoceglia/Documents/programmation/code_paper_mhe/results/img/frame_{frame}_{str(cond).replace('.','')}_{i}.svg"

        ax.set_ylabel(y_labels[i], fontsize=fontsize)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if i in [4, 5]:
            ax.set_xlabel(x_label)
            if c == 0 and f == 0:
                ax.set_xticks(x_ticks)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

    # set the size of the window
    fig.set_size_inches(10, 10)
    plt.tight_layout()
    svg = skunk.insert(skunk_dic)
    with open('fig_frame_errors.svg', 'w') as f:
        f.write(svg)
    # plt.show()
