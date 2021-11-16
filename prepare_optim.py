import biorbd
import bioviz
from biosiglive.client import Client
import numpy as np
from biosiglive.server import Server
from time import sleep
from biosiglive.live_mvc import ComputeMvc
from pyomeca import Markers, Analogs


def EKF(model, animate=True, return_markers_info=False):
    # for i in range(50):
        # client = Client(server_ip, server_port, "TCP")
        # markers_tmp = client.get_data(
        #     ["markers"], Nmhe=10, exp_freq=100, nb_of_data=10, get_names=True
        # )  # , get_kalman=False)
        # sleep((1 / 100) * 10)
        # if i == 0:
        #     mark0 = markers_tmp["markers"]
        #     markers = np.array(mark0).reshape((3, 6, 10))
        # #
        # else:
        #     mark_tmp = markers_tmp["markers"]
        #     mark_tmp = np.array(mark_tmp).reshape((3, 6, 10))
        #     markers = np.append(markers, mark_tmp, axis=2)

    # --- Markers --- #
    markers_full_names = ["Amedeo:XIPH", "Amedeo:STER", "Amedeo:T10", "Amedeo:CLAV_SC",
                          "Amedeo:CLAV_AC", "Amedeo:SCAP_AA", "Amedeo:SCAP_TS", "Amedeo:SCAP_IA",
                          "Amedeo:DELT",
                          "Amedeo:ARMl", "Amedeo:EPICl", "Amedeo:EPICm", "Amedeo:LARM_elb", "Amedeo:STYLr",
                          "Amedeo:STYLu"]
    #, "Amedeo:T1", "Amedeo:SCAP_CP"

    data_path = "data_wu_model/abd.c3d"
    markers_full = Markers.from_c3d(data_path, usecols=markers_full_names)
    marker_rate = int(markers_full.rate)
    marker_exp = markers_full[:, :, :].data * 1e-3
    markers = np.nan_to_num(marker_exp)
    muscle_names = ["Sensor 1.IM EMG1", "Sensor 10.IM EMG10", "Sensor 2.IM EMG2", "Sensor 3.IM EMG3",
                    "Sensor 4.IM EMG4",
                    "Sensor 5.IM EMG5", "Sensor 6.IM EMG6", "Sensor 7.IM EMG7", "Sensor 8.IM EMG8", "Sensor 9.IM EMG9"]

    # --- MVC --- #
    mvc_list = ["MVC_BIC", "MVC_DA", "MVC_DP", "MVC_LAT", "MVC_PEC", "MVC_TI", "MVC_TM", "MVC_TRI",
                "MVC_TS"]  # "MVC_DM",

    mvc_list_max = np.ndarray((len(muscle_names), 2000))
    mvc_list_val = np.ndarray((len(muscle_names), 2))
    for i in range(len(mvc_list)):
        b = Analogs.from_c3d(f"data_wu_model/{str(mvc_list[i])}.c3d", usecols=muscle_names)
        mvc_temp = (
            b.meca.band_pass(order=4, cutoff=[10, 425])
                .meca.center()
                .meca.abs()
                .meca.low_pass(order=4, cutoff=5, freq=b.rate)
            # .meca.normalize()
        )
        for j in range(len(muscle_names)):
            mvc_list_val = 1
        mvc_temp = -np.sort(-mvc_temp.data, axis=1)
        if i == 0:
            mvc_list_max = mvc_temp[:, :2000]
        else:
            mvc_list_max = np.concatenate((mvc_list_max, mvc_temp[:, :2000]), 1)

    mvc_list_max = -np.sort(-mvc_list_max, 1)[:, :2000]
    mvc_list_max = np.mean(mvc_list_max, 1)

    # data_path = "abd_co"
    a = Analogs.from_c3d(data_path, usecols=muscle_names)
    emg_rate = int(a.rate)
    emg = (
        a.meca.band_pass(order=4, cutoff=[10, 425])
            .meca.center()
            .meca.abs()
            .meca.low_pass(order=4, cutoff=5, freq=a.rate)
        # .meca.normalize()
    )
    # sio.savemat(f"./sujet_{sujet}/mvc_sujet_{sujet}.mat", {'mvc_treat': mvc_list_max})

    emg_norm_tmp = np.ndarray((len(muscle_names), emg.shape[1]))
    emg_norm = np.zeros((34, emg.shape[1]))
    for i in range(len(muscle_names)):
        emg_norm_tmp[i, :] = emg[i, :] / mvc_list_max[i]

    emg_norm[18, :] = emg_norm_tmp[0, :]
    emg_norm[3, :] = emg_norm_tmp[1, :]
    emg_norm[4, :] = emg_norm_tmp[2, :]
    emg_norm[21, :] = emg_norm_tmp[3, :]
    emg_norm[24, :] = emg_norm_tmp[4, :]
    emg_norm[[25, 26], :] = emg_norm_tmp[5, :]
    emg_norm[2, :] = emg_norm_tmp[6, :]
    emg_norm[[0, 1, 17], :] = emg_norm_tmp[7, :]
    emg_norm[[19, 20], :] = emg_norm_tmp[8, :]
    emg_norm[[11, 12, 13], :] = emg_norm_tmp[9, :]

        # marker_names = markers_tmp["marker_names"]
    # markers = markers #"* 0.001
    # marker_names = ["acromion", "CLAVSC", "EPICl", "EPICm", "STYLr", "STYLu"]
    # for i in range(4):
    #     data = client.get_data(data=["markers"], Nmhe=100, nb_of_data=100, exp_freq=100, get_names=True)
    # markers, marker_names = data['markers'], data["marker_names"]
    # markers = np.array(markers)
    # q_recons = data["kalman"]
    # from data_processing import read_data
    # file_name = f"/home/amedeo/Documents/programmation/RT_Optim/Solution/test_final_cmbbe/Results_MHE_markers_EMG_act_torque_driven_test_20210813-0830"
    # mat = read_data(file_name)
    # bmodel = biorbd.Model("/home/amedeo/Documents/programmation/RT_Optim/arm26_6mark_EKF.bioMod")
    # from pyomeca import Markers
    # markers = mat["kin_target"]
    bmodel = biorbd.Model(model)
    q_recons, a = Server.kalman_func(markers, model=bmodel)

    # -- modif model -- #
    # rot = np.array([q_recons[3, 0], q_recons[4, 0], q_recons[5, 0]])
    # trans = np.array([q_recons[0, 0], q_recons[1, 0], q_recons[2, 0]])
    # rt_thorax = biorbd.RotoTrans_fromEulerAngles(rot=rot, trans=trans, seq="xyz")
    # rot = np.array([q_recons[6, 0], q_recons[7, 0], q_recons[8, 0]])
    # trans = bmodel.localJCS(1).trans().to_array()
    # rt_hum = biorbd.RotoTrans_fromEulerAngles(rot=rot, trans=trans, seq="yzy")
    # rt_uln = bmodel.localJCS(5)
    # mod_model = BiorbdModel()
    # mod_model.read(model)
    # mod_model.segments[0].set_rot_trans_matrix(rt_thorax.to_array())
    # mod_model.segments[1].set_rot_trans_matrix(rt_hum.to_array())
    # # mod_model.write("mod_model.bioMod")
    # for i in range(bmodel.nbMarkers()):
    #     new_pos = markers[:, i, 0]
    #     if i in [0, 1]:
    #         bmodel.marker(i).applyRT(rt_thorax)
    #         bmodel.marker(i).setPosition(new_pos)
    #         bmodel.marker(i).applyRT(rt_thorax.transpose())
    #     elif i in [2, 3]:
    #         bmodel.marker(i).applyRT(rt_thorax)
    #         bmodel.marker(i).applyRT(rt_hum)
    #         bmodel.marker(i).setPosition(new_pos)
    #         bmodel.marker(i).applyRT(rt_hum.transpose())
    #         bmodel.marker(i).applyRT(rt_thorax.transpose())
    #     elif i in [4, 5]:
    #         bmodel.marker(i).applyRT(rt_thorax)
    #         bmodel.marker(i).applyRT(rt_hum)
    #         bmodel.marker(i).applyRT(rt_uln)
    #         bmodel.marker(i).setPosition(new_pos)
    #         bmodel.marker(i).applyRT(rt_uln.transpose())
    #         bmodel.marker(i).applyRT(rt_hum.transpose())
    #         bmodel.marker(i).applyRT(rt_thorax.transpose())
    #     mod_model.markers[i].set_position(bmodel.marker(i).to_array())
    # for i in
    # # mod_model.markers[0].set_position([0,1,2])
    # mod_model.write("mod_model_tmp.bioMod")

    if animate is True:
        b = bioviz.Viz(loaded_model=bmodel)
        # for i in range(q_recons.shape[1]):
        b.load_c3d(data_path)
        # b.load_experimental_markers(markers)
        # b.load_experimental_markers(Markers(markers))
        # b.load_movement(q_recons[:bmodel.nbQ(), i:i+1])
        # import time
        # time.sleep(1)
        # b.load_movement(q_recons[:bmodel.nbQ(), :])

        b.exec()
            # b.set_q(q_recons[:bmodel.nbQ(), i])
            # b.update()

    import scipy.io as sio
    sio.savemat("test_abd.mat", {"markers":markers, "emg": emg_norm, "kalman": q_recons})
    if return_markers_info is True:
        return q_recons, markers, marker_names
    else:
        return q_recons


if __name__ == "__main__":
    server_ip = 'localhost'
    # server_ip = "192.168.1.211"
    server_port = 50000
    muscle_range = (0, 1)
    # model_kalman_path = "/home/amedeo/Documents/programmation/RT_Optim/scaling_wu_model/wu_scaled_converted.bioMod"
    model_kalman_path = "wu_model.bioMod"
    # model_kalman_path = "arm26_6mark_scaled.bioMod"# Model with 6dofs on root
    # sleep(5)
    # reconstruct model:
    Q, markers, marker_names = EKF(model_kalman_path, animate=True, return_markers_info=True)

    MVC = ComputeMvc(
        muscle_range,
        acquisition_rate=100,
        frequency=2000,
        stream_mode="server_data",
        output_dir="test_08_12",
        output_file="mvc_prepare_optim.mat",
        server_port=server_port,
        server_ip=server_ip,
    )

    list_mvc = MVC.run()  # b, show_data=True)
    print(list_mvc)
