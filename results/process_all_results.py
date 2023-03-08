import numpy as np
from biosiglive.processing.data_processing import OfflineProcessing
import math
import biorbd
from biosiglive.file_io.save_and_load import save, load


# --- RMSE --- #
def rmse(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).mean())


def std(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).std())


def get_muscular_torque(x, act, model):
    """
    Get the muscular torque.
    """
    muscular_torque = np.zeros((model.nbQ(), x.shape[1]))
    states = model.stateSet()  # Get the muscle state set
    for i in range(act.shape[1]):
        for a, state in zip(act[:, i], states):
            state.setActivation(a)  # And fill it with the current value
        muscular_torque[:, i] = model.muscularJointTorque(
            states, x[: model.nbQ(), i], x[model.nbQ() : model.nbQ() * 2, i]
        ).to_array()
    return muscular_torque


def finite_difference(data, f):
    t = np.linspace(0, data.shape[0] / f, data.shape[0])
    y = data
    dydt = np.gradient(y, t)
    return dydt


def get_id_torque(x, model, f=33):
    q = x[: model.nbQ(), :]
    qdot = x[model.nbQ() : model.nbQ() * 2, :]
    qddot = np.zeros((q.shape[0], q.shape[1]))
    for i in range(q.shape[0]):
        qdot[i, :] = finite_difference(q[i, :], f)
        qddot[i, :] = finite_difference(qdot[i, :], f)
    qddot = OfflineProcessing().butter_lowpass_filter(qddot, 2, f, 4)

    tau_from_b = np.zeros((model.nbQ(), q.shape[1]))
    for i in range(q.shape[1]):
        tau_from_b[:, i] = model.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i]).to_array()
    return tau_from_b


if __name__ == "__main__":
    subject = "subject"
    conditions = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
    trials = [
        "data_abd_sans_poid",
        "data_abd_poid_2kg",
        "data_flex_poid_2kg",
        "data_cycl_poid_2kg",
        "data_flex_sans_poid",
        "data_cycl_sans_poid",
    ]
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
    result_mat = []
    t_est = []
    t_ref = []
    nb_mhe = []
    muscle_torque = []
    id_torque = []
    n_frames = [0, 25, 50, 75, 100]
    result_all_dic = {}
    n_init = [int(0)] * len(conditions)
    for trial in trials:
        if "2kg" in trial:
            model = biorbd.Model(f"wu_scaled_2kg.bioMod")
        else:
            model = biorbd.Model(f"wu_scaled.bioMod")
        result_dic = {}
        for c, cond in enumerate(conditions):
            result_dic_tmp = {}
            for f, frame in enumerate(n_frames):
                file = f"{trial}_result_duration_{cond}"
                result_mat = read_data("results_w6_freq/" + file)
                nb_mhe = int(result_mat["Nmhe"][0] + 1)
                rmse_markers = []
                rmse_torque = []
                std_markers = []
                std_torque = []
                result_mat["magnitude_emg_err"] = 0
                result_mat["phase_emg_err"] = 0
                result_mat["magnitude_emg_std"] = 0
                result_mat["phase_emg_std"] = 0
                nb_iter = len(result_mat["time"]) - n_init[c] - 1
                exp_freq = result_mat["exp_freq"][0]
                result_mat["U_est"] = result_mat["U_est"].clip(min=0.00000000000001)
                result_mat["U_est"] = result_mat["U_est"].clip(max=0.99999999999999)
                t_est = np.linspace(0, 100, nb_iter + 1)
                t_ref = np.linspace(0, 100, nb_iter + 1)
                n_frame = int((nb_mhe - 1) * frame / 100)
                if frame == 100:
                    n_frame = nb_mhe - 2
                x_int = np.zeros((model.nbQ() * 2, nb_iter))
                result_mat["ID_torque"] = get_id_torque(result_mat["X_est"][:, n_frame::nb_mhe][:, n_init[c] :], model)
                result_mat["muscle_torque"] = get_muscular_torque(
                    result_mat["X_est"][:, n_frame::nb_mhe][:, n_init[c] :],
                    result_mat["U_est"][:, n_frame::nb_mhe][:, n_init[c] :],
                    model,
                )
                result_mat["tau_est"] = result_mat["tau_est"][:, n_frame::nb_mhe][:, n_init[c] :]
                result_mat["est_tau_tot"] = result_mat["muscle_torque"] + result_mat["tau_est"]
                result_mat["X_est"] = result_mat["X_est"][:, n_frame::nb_mhe][:, n_init[c] :]
                result_mat["kalman"] = result_mat["kalman"][:, n_frame::nb_mhe][:, n_init[c] :]
                result_mat["U_est"] = result_mat["U_est"][:, n_frame::nb_mhe][:, n_init[c] :]
                result_mat["muscles_target"] = result_mat["muscles_target"][:, n_frame::nb_mhe][:, n_init[c] :]
                result_mat["kin_target"] = result_mat["kin_target"][:, :, n_frame::nb_mhe][:, :, n_init[c] :]
                result_mat["sol_freq_mean"] = np.mean(result_mat["sol_freq"])
                markers = np.ndarray((3, model.nbMarkers(), result_mat["X_est"].shape[1]))
                for i in range(result_mat["X_est"].shape[1]):
                    markers[:, :, i] = np.array(
                        [mark.to_array() for mark in model.markers(result_mat["X_est"][:, i])]
                    ).T
                for i in range(5, int(result_mat["X_est"].shape[0] / 2) - 1):
                    rmse_torque.append(
                        np.sqrt(np.mean((result_mat["est_tau_tot"][i, :] - result_mat["ID_torque"][i, :]) ** 2))
                    )
                    std_torque.append(
                        np.sqrt(np.std((result_mat["est_tau_tot"][i, :] - result_mat["ID_torque"][i, :]) ** 2))
                    )
                result_mat["saturation"] = np.where(result_mat["U_est"] > 0.95)[0].shape[0] * 100 / (result_mat["U_est"].shape[1]*result_mat["U_est"].shape[0])
                result_mat["gradient"] = np.sum(np.abs(np.gradient(result_mat["U_est"])))
                result_mat["rmse_torque"] = np.mean(rmse_torque)
                result_mat["std_torque"] = np.mean(std_torque)
                for m in range(model.nbMuscles()):
                    if m in muscle_track_idx:
                        idx = muscle_track_idx.index(m)
                        dt = (t_est[-1] - t_est[0]) / len(t_est)
                        vmm = (1 / (t_est[-1] - t_est[0])) * np.sum(result_mat["muscles_target"][idx, :] ** 2 * dt)
                        vcc = (1 / (t_est[-1] - t_est[0])) * np.sum(result_mat["U_est"][m, :] ** 2 * dt)
                        vcm = (1 / (t_est[-1] - t_est[0])) * np.sum(
                            result_mat["muscles_target"][idx, :] * result_mat["U_est"][m, :] * dt
                        )
                        M = np.sqrt(vcc / vmm) - 1
                        P = (1 / np.pi) * math.acos(vcm * (np.sqrt(vcc / vmm)))
                        result_mat["magnitude_emg_err"] += M
                        result_mat["phase_emg_err"] += P
                for i in range(model.nbMarkers()):
                    rmse_markers.append(
                        np.sqrt(np.mean((markers[0, i, :] - result_mat["kin_target"][0, i, :]) ** 2))
                        + np.sqrt(np.mean((markers[1, i, :] - result_mat["kin_target"][1, i, :]) ** 2))
                        + np.sqrt(np.mean((markers[2, i, :] - result_mat["kin_target"][2, i, :]) ** 2))
                    )
                result_mat["rmse_markers"] = np.mean(rmse_markers)
                result_mat["x_int"] = x_int
                result_mat["t_ref"] = t_ref
                result_mat["t_est"] = t_est
                result_mat["elevation"] = result_mat["X_est"][6, :]
                result_dic_tmp[f"{frame}"] = result_mat
            result_dic[f"{cond}"] = result_dic_tmp
        result_all_dic[f"{trial}"] = result_dic
        dic_to_save = {f"{trial}": result_all_dic[f"{trial}"]}
        add_data_to_pickle(dic_to_save, "result_all_trials")
