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
    id = False
    plot = True
    ik = False
    subject = "subject_3"
    trial = "data_abd_poid_2kg_test"
    data_dir = f"data_final_new/{subject}"
    osim_model_path = f"{data_dir}/wu_scaled.osim"
    trc_file = f"{data_dir}/{trial}.trc"
    # Torque from muscle activations
    estimated = read_data(f"{data_dir}/data_abd_poid_2kg_test_result_frame_0")
    nb_q = 10
    # activation = estimated["U_est"]
    activation = np.zeros((31, estimated["U_est"].shape[1]))
    q_est = estimated["X_est"][:nb_q, :]
    qdot_est = estimated["X_est"][nb_q:, :]
    qdot_est_real = estimated["X_est"][nb_q:, :]
    qdot_dif = np.copy(qdot_est)
    dofs = ["Sterno-claviculaire rot_x",
            "Sterno-claviculaire rot_y",
            "Acromio-claviculaire rot_x",
            "Acromio-claviculaire rot_y",
            "Acromio-claviculaire rot_z",
            "Gleno-humeral rot_y",
            "Gleno-humeral rot_z",
            "Gleno-humeral rot_y",
            "Coude rot_z"
            "coude rot_y"]

    for i in range(qdot_est.shape[0]):
        qdot_dif[i, :] = finite_difference(q_est[i, :], 70)
    if bpackage:
        bmodel_path = f"data_final_new/{subject}/wu_scaled.bioMod"
        # bmodel_path = f"data_final/{subject}/wu_wt_wrapp.bioMod"
        # bmodel_path = f"data_final_new/{subject}/Wu_Shoulder_Model_mod_wt_wrapp_Clara_scaled.bioMod"
        # bmodel_path = f"data_final/{subject}/Wu_Shoulder_Model.bioMod"

        model = biorbd.Model(bmodel_path)
        # for i in range(model.nbMuscles()):
        #     model.muscle(i).characteristics().setForceIsoMax(model.muscle(i).characteristics().forceIsoMax() * 2)

        # Muscular torque
        muscular_torque = np.zeros((nb_q, q_est.shape[1]))
        states = model.stateSet()  # Get the muscle state set
        for i in range(activation.shape[1]):
            for a, state in zip(activation[:, i], states):
                state.setActivation(a)  # And fill it with the current value
            muscular_torque[:, i] = model.muscularJointTorque(states, q_est[:, i], qdot_est_real[:, i]).to_array()
        if "tau_est" in estimated.keys():
            # for i in range(q_est.shape[0]):
            estimated_torque = muscular_torque + estimated["tau_est"]
        else:
            estimated_torque = muscular_torque
    # Torque from ID
    data = read_data(f"{data_dir}/{trial}")
    q = data["kalman"][:, ::3][:, :q_est.shape[1]]
    t = np.linspace(0, q.shape[1]/100, q.shape[1])
    if bpackage:
        q, qdot = kalman_func(data["markers"], model)
        # q, qdot = estimated["kalman"], qdot_est
        qddot = np.copy(q)
        # qdot = OfflineProcessing().butter_lowpass_filter(qdot, 5, 100, 2)
        for i in range(qdot.shape[0]):
            qdot[i, :] = finite_difference(q[i, :], 100)
        qdot = OfflineProcessing().butter_lowpass_filter(qdot, 5, 100, 2)
        for i in range(qdot.shape[0]):
            qddot[i, :] = finite_difference(qdot[i, :], 100)

        tau_from_b = np.zeros((model.nbQ(), q.shape[1]))
        for i in range(q.shape[1]):
            tau_from_b[:, i] = model.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i]).to_array()

    if opackage and id:
        # ID
        markers = data["markers"]
        marker_names = [
            "STER",
            "XIPH",
            "C7",
            "T10",
            "CLAV_SC",
            "CLAV_AC",
            "SCAP_IA",
            "Acrom",
            "SCAP_AA",
            "EPICl",
            "EPICm",
            "DELT",
            "ARMl",
            "STYLu",
            "LARM_elb",
            "STYLr",
        ]
        C3DtoTRC.WriteTrcFromMarkersData(
            trc_file,
            markers=markers[:3, :, :],
            marker_names=marker_names,
            data_rate=100,
            cam_rate=100,
            n_frames=markers.shape[2],
            units="m",
        ).write()

        # inverse kinematics for mot file
        model = osim.Model(osim_model_path)
        mot_output = f"{data_dir}/ik"
        # if ik:
        #     osim.InverseKinematicsTool().printToXML(f"{data_dir}/inverse_kin.xml")
        #     ik_input = f"{data_dir}/inverse_kin.xml"
        #     ik_output = f"{data_dir}/inverse_kin_out.xml"
        #     pyosim.InverseKinematics(osim_model_path, ik_input, ik_output, trc_file, mot_output)


        q_tmp = q
        q = np.zeros((17, q.shape[1]))
        for i in range(17):
            if i < 6 or i == 8 or i == 16:
                pass
            elif i > 8:
                q[i, :] = q_tmp[i - 1 - 6, :]
            else:
                q[i, :] = q_tmp[i - 6, :]

        write_mot(q, f"{mot_output}/{trial}.mot")

        # Disable muscle forces for inverse dynamics
        for i in range(model.getMuscles().getSize()):
            model.getMuscles().get(i).set_appliesForce(False)

        osim.InverseDynamics(model).printToXML("ID_tool.xml")
        pyosim.InverseDynamics(
            model,
            "ID_tool.xml",
            f"{data_dir}/ID_tool",
            f"{mot_output}/{trial}.mot",
            data_dir,
            sto_file_output=f"inverse_dynamics",
            low_pass=6,
            multi=False,
        )

    data = read_sto_mot_file(f"{data_dir}/inverse_dynamics.sto")
    data.pop("sternoclavicular_r3_moment")
    data.pop("time")
    tau = np.zeros((len(data.keys()), len(t)+1))
    for i, key in enumerate(data.keys()):
        if key != "time":
            tau[i, :] = data[key]

    data = read_sto_mot_file(f"{data_dir}/ik/{trial}.mot")
    q_osim = np.zeros((len(data.keys())-1, len(t)))
    data.pop("sternoclavicular_r3")
    data.pop("time")
    for i, key in enumerate(data.keys()):
        # if key != "time" and key != "sternoclavicular_r3":
        q_osim[i, :] = data[key]

    # b = bioviz.Viz(model_path=bmodel_path)
    # b.load_movement(q_est)
    # b.exec()
    if plot:
        plt.figure("q_dot")
        #plot
        t_est = np.linspace(0, t[-1], q_est.shape[1])
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(t_est, qdot_dif[i, :] * (180 / np.pi), label="qdot_from_q_est")
            plt.plot(t_est, qdot_est[i, :] * (180 / np.pi), label="qdot_est")
            # plt.plot(t, qdot[i, :] * 180 / np.pi, label="qdotkalman")
            plt.title(model.nameDof()[i].to_string())
        plt.legend()

        plt.figure("q")
        #plot
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(t[:], q_osim[6 + i, :], label="qosim")
            plt.plot(t_est, q_est[i, :] * (180 / np.pi), label="q_est")
            # plt.plot(t, q[i, :] * (180 / np.pi), label="qkalman")
            plt.title(model.nameDof()[i].to_string())
        plt.legend()

        plt.figure("Tau")
        #plot
        t_est = np.linspace(0, t[-1], q_est.shape[1])
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(t, tau[6 + i, :-1], label="osim")
            # plt.plot(t, tau_from_b[i, :], 'r', label="Couples par dynamique inverse (biorbd)")
            plt.plot(t_est, estimated_torque[i, :], label="Couples éstimés")
            plt.plot(t_est,muscular_torque[i, :], '--', label="Couples provenant des forces musculaires éstimés", )
            plt.plot(t_est,estimated["tau_est"][i, :],'-.', label="Couples résiduels éstimés")
            # plt.plot(t_est, estimated["tau_est"][i, :], label="residual")
            # plt.title(model.nameDof()[i].to_string())
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