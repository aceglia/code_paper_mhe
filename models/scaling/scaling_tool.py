# import biorbd
# from updated.ConvertOsim2Biorbd import ConvertedFromOsim2Biorbd4
import numpy as np
import csv
import scipy.io as sio
try:
    import pyosim
except ModuleNotFoundError:
    pass
try:
    import biorbd
except ModuleNotFoundError:
    pass


def _prepare_trc(file_name, units, marker_names, data_rate, cam_rate, n_frames, start_frame):
    headers = [
        ["PathFileType", 4,	"(X/Y/Z)", file_name],
        ["DataRate",
         "CameraRate",
         "NumFrames",
         "NumMarkers",
         "Units",
         "OrigDataRate",
         "OrigDataStartFrame",
         "OrigNumFrames"],
        [data_rate,	cam_rate, n_frames, len(marker_names), units,	data_rate, start_frame, n_frames]
    ]
    markers_row = ["Frame#", "Time", ]
    coord_row = ["", ""]
    empty_row = []
    idx = 0
    for i in range(len(marker_names)*3):
        if i % 3 == 0:
            markers_row.append(marker_names[idx])
            idx += 1
        else:
            markers_row.append(None)
    headers.append(markers_row)

    for i in range(len(marker_names)):
        name_coord = 0
        while name_coord < 3:
            if name_coord == 0:
                coord_row.append(f"X{i+1}")
            elif name_coord == 1:
                coord_row.append(f"Y{i+1}")
            elif name_coord == 2:
                coord_row.append(f"Z{i+1}")
            name_coord += 1

    headers.append(coord_row)
    headers.append(empty_row)
    return headers


def write_trc(file_name, markers, marker_names, data_rate, cam_rate, n_frames, start_frame=1, units="m"):
    headers = _prepare_trc(file_name, units, marker_names, data_rate, cam_rate, n_frames, start_frame)
    duration = n_frames / data_rate
    time = np.around(np.linspace(0, duration, n_frames), decimals=4)
    for frame in range(len(markers[0][0])):
        row = [frame + 1, time[frame]]
        for i in range(len(markers[0])):
            for j in range(3):
                row.append(markers[j][i][frame])
        headers.append(row)
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(headers)


if __name__ == '__main__':
    # Prepare trc file
    file_name = "markers_scaling.trc"
    data_rate = 100
    cam_rate = 100

    markers_full_names = ["Amedeo:XIPH", "Amedeo:STER", "Amedeo:T10", "Amedeo:CLAV_SC", "Amedeo:CLAV_AC",
                          "Amedeo:SCAP_AA", "Amedeo:SCAP_TS", "Amedeo:SCAP_IA", "Amedeo:DELT",
                          "Amedeo:ARMl", "Amedeo:EPICl", "Amedeo:EPICm", "Amedeo:LARM_elb", "Amedeo:STYLr",
                          "Amedeo:STYLu"]
    marker_names = ["XIPH", "STER", "T10", "CLAV_SC", "CLAV_AC",
                          "SCAP_AA", "SCAP_TS", "SCAP_IA", "DELT",
                          "ARMl", "EPICl", "EPICm", "LARM_elb", "STYLr",
                          "STYLu"]
    mat_content = sio.loadmat("/home/amedeo/Documents/programmation/RT_Optim/data_wu_model/test_abd.mat")

    markers = mat_content["kin_target"]
    n_frames = markers.shape[2]
    write_trc(file_name, markers, marker_names, data_rate, cam_rate, n_frames, units='m')
    #
    # stream marker position
    # sleep(10)
    # for i in range(4):
    #     client = Client(host_ip, host_port)
    #     markers_tmp = client.get_data(["markers"], Nmhe=nmhe, exp_freq=data_rate, get_names=True)
    #     sleep((1/data_rate)*nmhe)
    #     if i == 0:
    #         mark0 = markers_tmp["markers"]
    #         markers = np.array(mark0).reshape((3, 6, nmhe+1))
    #     else:
    #         mark_tmp = markers_tmp["markers"]
    #         mark_tmp = np.array(mark_tmp).reshape((3, 6, nmhe+1))
    #         markers = np.append(markers, mark_tmp, axis=2)
    #     # marker_names = markers_tmp["marker_names"]

    # -----SCALING----#
    # model_input = "arm26_6mark.osim"
    # model_output = "arm26_6mark_scaled.osim"
    # xlm_input = "scaling_tool.xml"
    # xlm_output = "scaled_file.xml"
    #
    # pyosim.Scale(model_input=model_input,
    #              model_output=model_output,
    #              xml_input=xlm_input,
    #              xml_output=xlm_output,
    #              static_path=file_name,
    #              mass=3.6,
    #              height=1
    #              )

# ------------ CONVERT ---------#
    # ConvertedFromOsim2Biorbd4("arm26_6mark_scaled.bioMod", "arm26_6mark_scaled.osim")


# --------KALMAN --------#
    model = biorbd.Model("arm26_6mark_scaled.bioMod")
    freq = 100  # Hz
    params = biorbd.KalmanParam(freq)
    kalman = biorbd.KalmanReconsMarkers(model, params)

    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model)
    markersOverFrames = []
    for i in range(markers.shape[2]):
        markersOverFrames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])

    q_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
    # print(q_recons)
    #

    # import bioviz
    # model = bioviz.Viz("arm26_6mark_scaled.bioMod")
    # model.exec()
    import bioviz
    b = bioviz.Viz(loaded_model=model)
    b.load_movement(q_recons)
    b.exec()