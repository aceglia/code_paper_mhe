"""
Script to scale the opensim model then translate it into biomod file and initialize it with a Kalman filter.
Data can be live-streamed or prerecorded to avoid the subject waiting.
"""

try:
    import biorbd
    import bioviz
except ModuleNotFoundError:
    pass
try:
    import pyosim
    import opensim
except ModuleNotFoundError:
    pass
import C3DtoTRC
import csv
from biosiglive.processing.msk_functions import kalman_func
from biosiglive.io.save_data import read_data, add_data_to_pickle


def read_sto_mot_file(filename):
    """
        Read sto or mot file from Opensim
        ----------
        filename: path
            Path of the file witch have to be read
        Returns
        -------
        Data Dictionary with file informations
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


def initialize(model_path: str, data_dir: str, scaling: bool = False, off_line: bool = True, mass: float = None):
    """
    Initialize the model with a Kalman filter and/or scale it.

    Parameters
    ----------
    model_path : str
        Path of the model to initialize
    data_dir : str
        Path of the directory where the data are stored
    scaling : bool, optional
        If True, the model will be scaled using opensim. The default is False.
    off_line : bool, optional
        If True, the model will be initialized and scaled with prerecorded data. The default is True.
    mass : int, optional
        Mass of the subject. The default is None.
    """
    mat = read_data(f"{data_dir}/C3D/{trial}")
    markers = mat["markers"][:3, :, :50]

    # Define the name of the model's markers
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
        "LARM_elb",
        "STYLr",
        "STYLu",
    ]

    # markers_tmp = np.copy(markers)
    # markers[:, 13, :] = markers_tmp[:, 14, :]
    # markers[:, 14, :] = markers_tmp[:, 15, :]
    # markers[:, 15, :] = markers_tmp[:, 13, :]

    if scaling:
        # ---------- model scaling ------------ #
        from pathlib import Path
        osim_model_path = model_path
        model_output = f"{data_dir}/" + Path(osim_model_path).stem + f"_scaled.osim"
        scaling_tool = f"{data_dir}/scaling_tool.xml"
        trc_file = f"{data_dir}/{trial}.trc"
        C3DtoTRC.WriteTrcFromMarkersData(
            trc_file,
            markers=markers[:3, :, :-1],
            marker_names=marker_names,
            data_rate=100,
            cam_rate=100,
            n_frames=markers.shape[2],
            units="m",
        ).write()

        # inverse kinematics for mot file
        opensim.InverseKinematicsTool().printToXML(f"{data_dir}/inverse_kin.xml")
        ik_input = f"{data_dir}/inverse_kin.xml"
        ik_output = f"{data_dir}/inverse_kin_out.xml"
        mot_output = f"{data_dir}/ik"
        pyosim.InverseKinematics(osim_model_path, ik_input, ik_output, trc_file, mot_output)

        pyosim.Scale(
            model_input=osim_model_path,
            model_output=model_output,
            xml_input=scaling_tool,
            xml_output=f"{data_dir}/scaling_tool_output.xml",
            static_path=trc_file,
            coordinate_file_name=f"{data_dir}/ik/{trial}.mot",
            mass=mass,
        )

        convert_model(
            in_path=f"{data_dir}/" + Path(model_output).stem + "_markers.osim",
            out_path=f"{data_dir}/" + Path(model_output).stem + ".bioMod",
            viz=False,
        )

    else:
        bmodel = biorbd.Model(model_path)
        q_recons, _ = kalman_func(markers, model=bmodel, use_kalman=True)
        q_mean = q_recons.mean(axis=1)
        print(q_mean[3], q_mean[4], q_mean[5]," xyz " , q_mean[0], q_mean[1], q_mean[2])
        b = bioviz.Viz(model_path=model_path)
        b.load_movement(q_recons) # Q from kalman array(nq, nframes)
        b.load_experimental_markers(markers) # expr markers array(3, nmarkers, nframes)
        b.exec()


def convert_model(in_path: str, out_path: str, viz: bool = None):
    """
    Convert a model from OpenSim to BioMod format.

    Parameters
    ----------
    in_path : str
        Path of the model to convert
    out_path : str
        Path of the converted model
    viz : bool, optional
        If True, the model will be visualized using bioviz package. The default is None.

    """
    #  convert_model
    from OsimToBiomod import Converter

    converter = Converter(out_path, in_path, muscle_type="Thelen2003Muscle")
    converter.main()
    if viz:
        b = bioviz.Viz(model_path=out_path)
        b.exec()


if __name__ == "__main__":
    trial = ["data_abd_sans_poid"]
        # "data_abd_poid_2kg",
        # "data_flex_sans_poid",

        #      "data_cycl_poid_2kg",
        #      "data_cycl_sans_poid",
        # "data_flex_poid_2k",


    # trial = trial[0]
    model_path = "data_final_new/subject_3/C3D/wu_scaled.bioMod"
    model = biorbd.Model(model_path)
    # for i in trial:
        # results_mat_abd = read_data(f"data_final_new/subject_3/{i}")
        # # results_mat_flex = read_data(f"data_final_new/subject_3/{trial}")
        # # convert_model(f"data_final_new/subject_3/wu_scaled.osim", f"data_final_new/subject_3/wu_scaled.bioMod", viz=True)
        # # model_path = f"{data_dir}/wu.osim"
        # # # model = biorbd.Model(model_path)
        #
        # q_recons, q_dot = kalman_func(results_mat_abd["markers"][:, :, 1000:], return_q_dot=True, model=model, use_kalman=True)
        # import numpy as np
        #
        # # def finite_difference(data, f):
        # #     t = np.linspace(0, data.shape[0] / f, data.shape[0])
        # #     y = data
        # #     dydt = np.gradient(y, t)
        # #     return dydt
        # import matplotlib.pyplot as plt
        # for s in range(len(trial)):
        #     plt.figure("markers")
        #     for i in range(model.nbMarkers()):
        #         plt.subplot(4, 4, i + 1)
        #         plt.plot(results_mat_abd["markers"][0, i, 1000:].T, "-r")
        #         plt.plot(results_mat_abd["markers"][1, i, 1000:].T, "-r")
        #         plt.plot(results_mat_abd["markers"][2, i, 1000:].T, "-r")
        # plt.show()
        # b = bioviz.Viz(model_path=model_path)
        # b.load_movement(q_recons)
        # b.load_experimental_markers(results_mat_abd["markers"][:, :, 1000:])
        # b.exec()
        # # q_dot = np.copy(q_recons)
        # # for j in range(q_recons.shape[0]):
        # #     q_dot[j, :] = finite_difference(q_recons[j, :], 100)
        # results_mat_abd["emg_proc"] = results_mat_abd["emg_proc"][:, 1000:]
        # results_mat_abd["markers"] = results_mat_abd["markers"][:, :, 1000:]
        # results_mat_abd["kalman"] = np.concatenate((q_recons, q_dot), axis=0)
        # add_data_to_pickle(results_mat_abd, f"{i}_test")


    # import matplotlib.pyplot as plt
    # for s in range(len(trial)):
    #     plt.figure("markers")
    #     for i in range(model.nbMarkers()):
    #         plt.subplot(4, 4, i + 1)
    #         plt.plot(results_mat_abd["markers"][0, i, :].T, "-r")
    #         plt.plot(results_mat_abd["markers"][1, i, :].T, ".-r")
    #         plt.plot(results_mat_abd["markers"][2, i, :].T, "--r")
    #         plt.plot(results_mat_flex["markers"][0, i, :].T, "-b")
    #         plt.plot(results_mat_flex["markers"][1, i, :].T, ".-b")
    #         plt.plot(results_mat_flex["markers"][2, i, :].T, "--b")
    # plt.show()
    subject = "subject_3"
    mass = 72
    mass_scaling = mass * 0.578 + mass * 0.050
    data_dir = f"data_final_new/{subject}"
    model_path = f"{data_dir}/C3D/wu_scaled.bioMod"
    # convert_model(f"{data_dir}/wu_scaled_markers.osim", f"{data_dir}/wu_scaled_markers.bioMod", viz=True)
    # model_path = f"{data_dir}/wu.osim"
    trial = trial[0]
    # model = biorbd.Model(model_path)
    # for i, muscle in enumerate(model.muscleNames()):
    #     print(f"idx: {i} - name: {muscle.to_string()}")
    initialize(model_path, data_dir, scaling=False, off_line=True, mass=mass_scaling)

    # Write file
    # data_path = f"{data_dir}/{trial}"
    # data_path_saved = f"{data_dir}/{trial}_w_dq_scaled"
    # data = read_data(data_path)
    # bmodel = biorbd.Model(model_path)
    # # b = bioviz.Viz(model_path=model_path)
    # # b.exec()
    # markers = data["markers"]
    # markers_tmp = np.copy(markers)
    # markers[:, 13, :] = markers_tmp[:, 14, :]
    # markers[:, 14, :] = markers_tmp[:, 15, :]
    # markers[:, 15, :] = markers_tmp[:, 13, :]
    # q_recons, q_dot_recons = kalman_func(markers, model=bmodel, return_q_dot=True, return_kalman=False)
    # # b = bioviz.Viz(model_path=model_path)
    # # b.load_movement(q_recons)
    # # b.load_experimental_markers(markers)
    # # b.exec()
    # x_kalman = np.concatenate((q_recons, q_dot_recons), axis=0)
    # import matplotlib.pyplot as plt
    # plt.figure("q")
    # for i in range(9):
    #     plt.subplot(3, 3, i + 1)
    #     plt.plot(x_kalman[i, :], label="kalman")
    # plt.figure("qdot")
    # for i in range(9, 18):
    #     plt.subplot(3, 3, i - 9 + 1)
    #     plt.plot(x_kalman[i, :], label="kalman")
    # # plt.show()
    # add_data_to_pickle({"kalman": x_kalman, "markers": data["markers"], "emg": data["emg"]}, data_path_saved)
    # # plot kalman
