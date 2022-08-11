import os.path

try:
    import biorbd
    import bioviz
    bpackage = True
except:
    bpackage = False
    pass
import numpy as np
import matplotlib.pyplot as plt
from biosiglive.io.save_data import read_data, add_data_to_pickle
from biosiglive.processing.msk_functions import kalman_func
try:
    import opensim as osim
    import pyosim
    opackage = True
except:
    opackage = False
    pass
import C3DtoTRC
import csv

from biosiglive.processing.data_processing import OfflineProcessing
from scipy.interpolate import interp1d


def headers_def(n_frame):
    return [["Coordinates"],
               ["version = 1"],
               [f"nRows = {n_frame}"],
               ["nColumns=18"],
               ["inDegrees=yes"],
               [],
               ["Units are S.I. units (second, meters, Newtons, ...)"],
               ["If the header above contains a line with 'inDegrees',"
                " this indicates whether rotational values are in degrees (yes) or radians (no)."],
               [],
               ["endheader"],
               ["time", "thorax_tilt", "thorax_list", "thorax_rotation", "thorax_tx", "thorax_ty",
                "thorax_tz", "sternoclavicular_r1", "sternoclavicular_r2", "sternoclavicular_r3",
                "Acromioclavicular_r1", "Acromioclavicular_r2", "Acromioclavicular_r3", "shoulder_plane",
                "shoulder_ele", "shoulder_rotation", "elbow_flexion", "pro_sup"]
               ]


def finite_difference(data, f):
    t = np.linspace(0, data.shape[0]/f, data.shape[0])
    y = data
    dydt = np.gradient(y, t)

    # data_dot = np.ndarray(data.shape)
    # data_dot[:] = np.nan
    #
    # for i in range(data_dot.shape[1] - 1):
    #     data_dot[:, i] = (data[:, i - 1] - data[:, i + 1]) / (2 * 0.01)

    return dydt


def write_mot(q, file_path):
    """
    Write the mot file.
    """
    n_frame = q.shape[1]
    duration = q.shape[1] / 100
    time = np.around(np.linspace(0, duration, n_frame), decimals=2)
    headers = headers_def(n_frame)
    for frame in range(q.shape[1]):
        row = [time[frame]]
        for i in range(q.shape[0]):
            row.append(np.round(q[i, frame] * 180/np.pi, 2))
        headers.append(row)
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows(headers)


def read_sto_mot_file(filename):
    """
        Read sto or mot file from Opensim
        ----------
        filename: str
            Path of the file witch have to be read
        Returns
        -------
        Data Dictionary with file information
    """
    data = {}
    data_row = []
    first_line = ()
    end_header = False
    with open(f"{filename}", "rt") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) == 0:
                pass
            elif row[0][:9] == "endheader":
                end_header = True
                first_line = idx + 1
            elif end_header is True and row[0][:9] != "endheader":
                row_list = row[0].split("\t")
                if idx == first_line:
                    names = row_list
                else:
                    data_row.append(row_list)

    for r in range(len(data_row)):
        for col in range(len(names)):
            if r == 0:
                data[f"{names[col]}"] = [float(data_row[r][col])]
            else:
                data[f"{names[col]}"].append(float(data_row[r][col]))
    return data


if __name__ == '__main__':
    plot = True
    subject = "subject_3"
    data_dir = f"data_final_new/{subject}"

    # Torque from muscle activations
    bmodel_path = f"data_final_new/{subject}/wu_scaled.bioMod"
    # bmodel_path = f"data_final_new/subject_1/model_subject_1_scaled.bioMod"
    model = biorbd.Model(bmodel_path)
    # model = biorbd.Model(f"data_final_new/subject_1/model_subject_1_scaled.bioMod")
    data = read_data(f"{data_dir}/data_abd_sans_poid_test")
    estimated = read_data(f"{data_dir}/data_abd_sans_poid_test_result_frame_008_0")
    # estimated = read_data(f"{data_dir}/abd_cocon_w_dq_result")

    activation = estimated["U_est"]
    nb_q = model.nbQ()
    q_est = estimated["X_est"][:nb_q, :]
    qdot_est = estimated["X_est"][nb_q:, :]
    qdot_dif = np.copy(qdot_est)
    dofs = ["Sterno-claviculaire rot_x",
            "Sterno-claviculaire rot_y",
            # "Sterno-claviculaire rot_z",
            "Acromio-claviculaire rot_x",
            "Acromio-claviculaire rot_y",
            "Acromio-claviculaire rot_z",
            "Gleno-humeral rot_y",
            "Gleno-humeral rot_z",
            "Gleno-humeral rot_y",
            "Coude rot_z",
            "Coude rot_y"]
    # Muscular torque
    muscular_torque = np.zeros((nb_q, q_est.shape[1]))
    states = model.stateSet()  # Get the muscle state set
    for i in range(activation.shape[1]):
        for a, state in zip(activation[:, i], states):
            state.setActivation(a)  # And fill it with the current value
        muscular_torque[:, i] = model.muscularJointTorque(states, q_est[:, i], qdot_est[:, i]).to_array()

    if "tau_est" in estimated.keys():
        estimated_torque = muscular_torque + estimated["tau_est"]
    else:
        estimated_torque = muscular_torque

    # Torque from ID
    # q, qdot = kalman_func(data["markers"][:, :, 350:-100], model, use_kalman=False)
    q = estimated["kalman"][:nb_q, :]
    qdot = estimated["kalman"][nb_q:nb_q * 2, :]
    # q = estimated["X_est"][:nb_q, :]
    # qdot = estimated["X_est"][nb_q:nb_q * 2, :]
    t = np.linspace(0, q.shape[1]/100, q.shape[1])
    # qdot = np.zeros((q.shape[0], q.shape[1]))
    qddot = np.zeros((q.shape[0], q.shape[1]))
    # qdot = OfflineProcessing().butter_lowpass_filter(qdot, 2, 100, 4)
    for i in range(q.shape[0]):
        qdot[i, :] = finite_difference(q[i, :], 100)
        # qdot[i, :] = OfflineProcessing().butter_lowpass_filter(qdot[i, :], 2, 100, 4)
        qddot[i, :] = finite_difference(qdot[i, :], 100)
    qddot = OfflineProcessing().butter_lowpass_filter(qddot, 2, 100, 4)

    tau_from_b = np.zeros((model.nbQ(), q.shape[1]))
    for i in range(q.shape[1]):
        tau_from_b[:, i] = model.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i]).to_array()
    # tau_from_b = OfflineProcessing().butter_lowpass_filter(qddot, 2, 100, 4)

    if plot:
        plt.figure("q_dot")
        #plot
        t_est = np.linspace(0, t[-1], q_est.shape[1])
        for i in range(nb_q):
            plt.subplot(4, 3, i + 1)
            plt.plot(t_est, qdot_est[i, :] * (180 / np.pi), label="qdot_est")
            plt.plot(t, estimated["kalman"][nb_q:nb_q * 2, :][i,:] * (180 / np.pi), label="qdotkalman")
            plt.title(model.nameDof()[i].to_string())
        plt.legend()

        plt.figure("q")
        for i in range(nb_q):
            plt.subplot(4, 3, i + 1)
            plt.plot(t_est, q_est[i, :] * (180 / np.pi), label="q_est")
            plt.plot(t, q[i, :] * (180 / np.pi), label="qkalman")
            plt.title(model.nameDof()[i].to_string())
        plt.legend()

        plt.figure("Tau")
        #plot
        t_est = np.linspace(0, t[-1], q_est.shape[1])
        for i in range(nb_q):
            plt.subplot(4, 3, i + 1)
            # plt.plot(t, tau[6 + i, :], label="osim")
            plt.plot(t, tau_from_b[i, :], 'r', label="Couples par dynamique inverse (biorbd)")
            plt.plot(t_est, estimated_torque[i, :], label="Couples éstimés")
            plt.plot(t_est, muscular_torque[i, :], '--', label="Couples provenant des forces musculaires éstimés", )
            plt.plot(t_est, estimated["tau_est"][i, :], '-.', label="Couples résiduels éstimés")
            plt.title(dofs[i])
            if i in [6,7,8]:
                plt.xlabel("Temps (s)")
            if i in [0, 3, 6]:
                plt.ylabel("Couple (N.m)")
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 0.80), loc="upper left", frameon=False, fontsize=12)


        def rmse(data, data_ref):
            return np.sqrt(((data - data_ref) ** 2).mean())


        def std(data, data_ref):
            return np.sqrt(((data - data_ref) ** 2).std())

        # rmse_torque = rmse(estimated_torque, tau_from_b[:,:636][:, ::3])
        # std_torque = std(estimated_torque, tau_from_b[:,:636][:, ::3])

        plt.show()