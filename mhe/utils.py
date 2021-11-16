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
    # conf = MuscleForceEstimator._check_and_adjust_dim(*args)
    if estimator_instance.data_to_show.count("force") != 0:
        estimator_instance.force_to_plot = np.append(estimator_instance.force_to_plot[:, -estimator_instance.exp_freq - 1:], force_est, axis=1)
        # if estimator_instance.count_p_f == estimator_instance.plot_force_ratio:
        update_plot_force(
            estimator_instance.force_to_plot, estimator_instance.p_force, estimator_instance.app_force, estimator_instance.plot_force_ratio, estimator_instance.muscle_names
        )  # , box_force)
        estimator_instance.count_p_f = 0
        # else:
        estimator_instance.count_p_f += 1

    if estimator_instance.data_to_show.count("q") != 0:
        # estimator_instance.q_to_plot = np.append(estimator_instance.q_to_plot[:, -estimator_instance.exp_freq - 1 :], q_est.reshape(-1, 1), axis=1)
        # # if estimator_instance.count_p_q == estimator_instance.plot_force_ratio:
        # update_plot_q(
        #     estimator_instance.q_to_plot * (180 / np.pi),
        #     estimator_instance.p_q,
        #     estimator_instance.app_q,
        #     estimator_instance.box_q
        # )
        # estimator_instance.count_p_q = 0
        # estimator_instance.b.load_experimental_markers(estimator_instance.kin_target[:, :, -1:])
        estimator_instance.wind.set_q(q_est)
        # estimator_instance.b.update()
        # else:
        #     estimator_instance.count_p_q += 1


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


def save_results(sol, data, current_time, kin_data_to_track, track_emg=False, use_torque=True, use_excitation=False):
    if use_torque:
        data["tau_est"] = sol.controls["tau"][:, -2:-1]
    dyn = "act" if use_excitation is not True else "exc"
    torque = "_torque" if use_torque else ""
    emg = "EMG_" if track_emg else ""
    result_dir = f"results_{strftime('%Y%m%d-%H%M')[:8]}"
    if not os.path.isdir(f"results/{result_dir}"):
        os.mkdir(f"results/{result_dir}")
    data_path = f"results/{result_dir}/Results_MHE_{kin_data_to_track}_{emg}{dyn}{torque}_driven_test_{current_time}"
    add_data_to_pickle(data, data_path)