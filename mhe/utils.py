import numpy as np
from biosiglive.data_processing import add_data_to_pickle
from biosiglive.data_plot import init_plot_force, init_plot_q, update_plot_force, update_plot_q
from time import strftime
import os


def check_and_adjust_dim(*args):
    if len(args) == 1:
        conf = args[0]
    else:
        conf = {}
        for i in range(len(args)):
            for key in args[i].keys():
                conf[key] = args[i][key]
    return conf


def update_plot(estimator_instance, t, force_est, q_est):
    if estimator_instance.data_to_show.count("force") != 0:
        estimator_instance.force_to_plot = np.append(estimator_instance.force_to_plot[:, -estimator_instance.exp_freq - 1:], force_est, axis=1)
        update_plot_force(
            estimator_instance.force_to_plot, estimator_instance.p_force, estimator_instance.app_force, estimator_instance.plot_force_ratio, estimator_instance.muscle_names
        )
        estimator_instance.count_p_f = 0
        estimator_instance.count_p_f += 1
    if estimator_instance.data_to_show.count("q") != 0:
        estimator_instance.b.set_q(np.array(q_est)[:, -1])


def compute_force(sol, get_force, nbMT, use_excitation=False):
    force_est = np.zeros((nbMT, 1))
    q_est = sol.states["q"][:, -2:-1]
    dq_est = sol.states["qdot"][:, -2:-1]
    if use_excitation:
        a_est = sol.states["muscles"][:, -1:]
        u_est = sol.controls["muscles"][:, -2:-1]
    else:
        a_est = sol.controls["muscles"][:, -2:-1]
        u_est = a_est
    # Compute force
    for i in range(nbMT):
        force_est[i, 0] = get_force(q_est, dq_est, a_est, u_est)[i, :]
    return force_est, q_est, dq_est, a_est, u_est


def save_results(sol,
                 data,
                 current_time,
                 kin_data_to_track,
                 track_emg=False,
                 use_torque=True,
                 use_excitation=False,
                 result_dir=None,
                 file_name=None,
                 is_mhe=True):
    if use_torque:
        data["tau_est"] = sol.controls["tau"][:, -2:-1]
    dyn = "act" if use_excitation is not True else "exc"
    torque = "_torque" if use_torque else ""
    emg = "_EMG_" if track_emg else "_"
    full = "full_" if not is_mhe else "mhe_"
    file_name = file_name if file_name else f"Results_{full}{kin_data_to_track}{emg}{dyn}{torque}_driven_{current_time}"
    result_dir = result_dir if result_dir else f"results/results_{strftime('%Y%m%d-%H%M')[:8]}"
    if not os.path.isdir(f"results/"):
        os.mkdir(f"results/")
    data_path = f"{result_dir}/{file_name}"
    add_data_to_pickle(data, data_path)