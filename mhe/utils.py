"""
This code provides some utility functions for the mhe implementation.
"""

import numpy as np
import bioptim
from biosiglive.io.save_data import add_data_to_pickle, read_data
from biosiglive.interfaces.client_interface import TcpClient
from biosiglive.streaming.client import Client
from biosiglive.gui.plot import LivePlot
from time import strftime
import datetime
from scipy.interpolate import interp1d
import os
import scipy.io as sio


def check_and_adjust_dim(*args):
    """
    Check if the dimensions of the arguments are the same.
    If not, the function will adjust the dimensions to be the same.
    """
    if len(args) == 1:
        conf = args[0]
    else:
        conf = {}
        for i in range(len(args)):
            for key in args[i].keys():
                conf[key] = args[i][key]
    return conf


def update_plot(estimator_instance, force_est: np.ndarray, q_est: np.ndarray, init_time: float = None):
    """
    Update the plot of the mhe.

    Parameters
    ----------
    estimator_instance: instance of the estimator class
        The estimator class.
    force_est: np.ndarray
        The estimated force.
    q_est: np.ndarray
        The estimated joint angles.
    init_time: float
        The initial time.
    """

    absolute_delay_plot = 0
    if estimator_instance.data_to_show.count("force") != 0:
        estimator_instance.force_to_plot = np.append(
            estimator_instance.force_to_plot[:, -estimator_instance.exp_freq - 1 :], force_est, axis=1
        )
        estimator_instance.all_plot.update_plot_window(
            estimator_instance.all_plot.plot[0],
            data=estimator_instance.force_to_plot,
            app=estimator_instance.app_force,
            rplt=estimator_instance.rplt_force,
            box=estimator_instance.layout_force,
        )

        estimator_instance.count_p_f = 0
        estimator_instance.count_p_f += 1

    if estimator_instance.data_to_show.count("q") != 0:
        n_plot = 0 if not "force" in estimator_instance.data_to_show else 1
        estimator_instance.all_plot.update_plot_window(estimator_instance.all_plot.plot[n_plot], np.array(q_est)[:, -1])

    if init_time:
        absolute_time_received = datetime.datetime.now()
        absolute_time_received_dic = {
            "day": absolute_time_received.day,
            "hour": absolute_time_received.hour,
            "hour_s": absolute_time_received.hour * 3600,
            "minute": absolute_time_received.minute,
            "minute_s": absolute_time_received.minute * 60,
            "second": absolute_time_received.second,
            "millisecond": int(absolute_time_received.microsecond / 1000),
            "millisecond_s": int(absolute_time_received.microsecond / 1000) * 0.001,
        }
        absolute_time_received_s = 0
        for key in absolute_time_received_dic.keys():
            if key == "second" or key[-1:] == "s":
                absolute_time_received_s = absolute_time_received_s + absolute_time_received_dic[key]
        absolute_delay_plot = absolute_time_received_s - init_time

    return np.round(absolute_delay_plot, 3)


def compute_force(
    sol: bioptim.Solution, get_force, nbmt: int, frame_to_save: int = 0, slide_size=1, save_all_frame=False
):
    """
    Compute the force.

    Parameters
    ----------
    sol: bioptim.Solution
        The solution of the mhe.
    get_force: function
        The function that computes the force.
    nbmt: int
        The number of muscles.
    use_excitation: bool
        If True, the excitation will be used.

    Returns
    -------
    Tuple of the force, joint angles, activation and excitation.
    """
    if frame_to_save >= sol.states["q"].shape[1] - 1 + slide_size:
        raise RuntimeError(
            f"You can ask to save frame from 0 to {sol.states['q'].shape[1] + slide_size}." f"You asked{frame_to_save}."
        )
    force_est = np.zeros((nbmt, slide_size))
    if not save_all_frame:
        q_est = sol.states["q"][:, frame_to_save : frame_to_save + slide_size]
        dq_est = sol.states["qdot"][:, frame_to_save : frame_to_save + slide_size]
        a_est = sol.controls["muscles"][:, frame_to_save : frame_to_save + slide_size]
    else:
        q_est = sol.states["q"]
        dq_est = sol.states["qdot"]
        a_est = sol.controls["muscles"]
    u_est = a_est
    for i in range(nbmt):
        for j in range(slide_size):
            force_est[i, j] = get_force(q_est[:, j], dq_est[:, j], a_est[:, j], u_est[:, j])[i, :]
    return q_est, dq_est, a_est, u_est, force_est


def save_results(
    data: dict,
    current_time: str,
    kin_data_to_track: str = "markers",
    track_emg: bool = False,
    use_torque: bool = True,
    result_dir: bool = None,
    file_name: bool = None,
    file_name_prefix: str = "",
):
    """
    Save the results.

    Parameters
    ----------
    data: dict
        The data to save.
    current_time: str
        The current time.
    kin_data_to_track: str
        The data to track.
    track_emg: bool
        If True, the emg have been tracked.
    use_torque: bool
        If True, the torque have been used.
    result_dir: bool
        The directory where the results will be saved.
    file_name: bool
        The name of the file where the results will be saved.
    file_name_prefix: str
        The prefix of the file name.
    """
    torque = "_torque" if use_torque else ""
    emg = "_EMG_" if track_emg else "_"
    file_name = file_name if file_name else f"Results_mhe_{kin_data_to_track}{emg}{torque}_driven_{current_time}"
    file_name = file_name_prefix + file_name
    result_dir = result_dir if result_dir else f"results/results_{strftime('%Y%m%d-%H%M')[:8]}"
    if not os.path.isdir(f"results/"):
        os.mkdir(f"results/")
    data_path = f"{result_dir}/{file_name}"
    add_data_to_pickle(data, data_path)


def muscle_mapping(muscles_target_tmp: np.ndarray, muscle_track_idx: list, mvc_list=None):
    """
    Map the muscles to the right index.

    Parameters
    ----------
    muscles_target_tmp: np.ndarray
        The muscles target.
    mvc_list: list
        The list of the mvc.
    muscle_track_idx: list
        The list of the muscle index.

    Returns
    -------
    The mapped muscles.
    """
    muscles_target = np.zeros((len(muscle_track_idx), int(muscles_target_tmp.shape[1])))
    muscles_target[[0, 1, 2], :] = muscles_target_tmp[0, :]
    muscles_target[[3], :] = muscles_target_tmp[1, :]
    muscles_target[4, :] = muscles_target_tmp[2, :]
    muscles_target[5, :] = muscles_target_tmp[3, :]
    muscles_target[[6, 7], :] = muscles_target_tmp[4, :]
    muscles_target[[8, 9, 10], :] = muscles_target_tmp[5, :]
    muscles_target[[11, 12], :] = muscles_target_tmp[6, :]
    muscles_target[[13], :] = muscles_target_tmp[7, :]
    muscles_target[[14], :] = muscles_target_tmp[8, :]
    muscles_target[[15], :] = muscles_target_tmp[9, :]
    if mvc_list:
        muscles_target = muscles_target / np.repeat(mvc_list, muscles_target_tmp.shape[1]).reshape(
            len(mvc_list), muscles_target_tmp.shape[1]
        )
    return muscles_target


def interpolate_data(interp_factor: int, x_ref: np.ndarray, muscles_target: np.ndarray, markers_target: np.ndarray):
    """
    Interpolate the reference and target data.

    Parameters
    ----------
    interp_factor: int
        The interpolation factor.
    x_ref: np.ndarray
        The reference x.
    muscles_target: np.ndarray
        The reference muscles.
    markers_target: np.ndarray
        The reference markers.

    Returns
    -------
    Tuple of interpolated data.
    """
    # interpolate target
    if interp_factor != 1:
        # x_ref
        x = np.linspace(0, x_ref.shape[1] / 100, x_ref.shape[1])
        f_x = interp1d(x, x_ref)
        x_new = np.linspace(0, x_ref.shape[1] / 100, int(x_ref.shape[1] * interp_factor))
        x_ref = f_x(x_new)

        # markers_ref
        markers_ref = np.zeros((3, markers_target.shape[1], int(markers_target.shape[2] * interp_factor)))
        for i in range(3):
            x = np.linspace(0, markers_target.shape[2] / 100, markers_target.shape[2])
            f_mark = interp1d(x, markers_target[i, :, :])
            x_new = np.linspace(0, markers_target.shape[2] / 100, int(markers_target.shape[2] * interp_factor))
            markers_ref[i, :, :] = f_mark(x_new)

        # muscle_target
        x = np.linspace(0, muscles_target.shape[1] / 100, muscles_target.shape[1])
        f_mus = interp1d(x, muscles_target)
        x_new = np.linspace(0, muscles_target.shape[1] / 100, int(muscles_target.shape[1] * interp_factor))
        muscles_target = f_mus(x_new)
    else:
        markers_ref = markers_target
    return x_ref, markers_ref, muscles_target


def get_data(ip=None, port=None, message=None, offline=False, offline_file_path=None):
    if offline:
        nfinal = -400
        if offline_file_path[-4:] == ".mat":
            mat = sio.loadmat(offline_file_path)
            x_ref, markers, muscles = mat["kalman"], mat["markers"], mat["emg_proc"]

        else:
            mat = read_data(offline_file_path)
            try:
                x_ref, markers, muscles = mat["kalman"], mat["kin_target"], mat["muscles_target"]
            except:
                x_ref, markers, muscles = (
                    mat["kalman"][:, 100:nfinal],
                    mat["markers"][:, 4:, 100:nfinal],
                    mat["emg_proc"][:, 100:nfinal],
                )
        return x_ref, markers, muscles
    else:
        client = Client(ip, port, "TCP")
        return client.get_data(message)
