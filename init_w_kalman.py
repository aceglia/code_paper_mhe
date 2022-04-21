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
from biosiglive.client import Client
from biosiglive.server import Server
from biosiglive.data_processing import read_data
import numpy as np
from time import sleep


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


def initialize(model_path: str, data_dir: str, scaling: bool = False, off_line: bool = True, mass: int = None):
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

    if not off_line:
        # Stream data from server and store them in an array
        server_ip = "192.168.1.211"
        server_port = 50000
        n_marks = 16
        for i in range(5):
            client = Client(server_ip, server_port, "TCP")
            markers_tmp = client.get_data(["markers"], read_frequency=100, nb_of_data_to_export=10, get_names=True)
            sleep((1 / 100) * 10)
            if i == 0:
                mark_0 = markers_tmp["markers"]
                marker_names = markers_tmp["marker_names"]
                markers = np.array(mark_0).reshape((3, n_marks, 10))
            else:
                mark_tmp = markers_tmp["markers"]
                mark_tmp = np.array(mark_tmp).reshape((3, n_marks, 10))
                markers = np.append(markers, mark_tmp, axis=2)
    else:
        mat = read_data(f"{data_dir}/{trial}")
        try:
            markers = mat["kin_target"][:3, :, :]
        except:
            markers = mat["markers"][:3, :, :]

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
        "STYLu",
        "LARM_elb",
        "STYLr",
    ]

    if scaling:
        # ---------- model scaling ------------ #
        from pathlib import Path

        osim_model_path = f"{data_dir}/model_{subject}.osim"
        model_output = f"{data_dir}/" + Path(osim_model_path).stem + f"_scaled.osim"
        scaling_tool = f"{data_dir}/scaling_tool.xml"
        trc_file = f"{data_dir}/anato.trc"
        C3DtoTRC.WriteTrcFromMarkersData(
            trc_file,
            markers=markers[:3, :, :20],
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
            coordinate_file_name=f"{data_dir}/ik/anato.mot",
            mass=mass,
        )

        convert_model(
            in_path=f"{data_dir}/" + Path(model_output).stem + "_markers.osim",
            out_path=f"{data_dir}/" + Path(model_output).stem + ".bioMod",
            viz=False,
        )

    else:
        bmodel = biorbd.Model(model_path)

        q_recons, _ = Server.kalman_func(markers, model=bmodel)
        b = bioviz.Viz(model_path=model_path)
        b.load_movement(q_recons)
        b.load_experimental_markers(markers)
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

    converter = Converter(out_path, in_path)
    converter.main()
    if viz:
        b = bioviz.Viz(model_path=out_path)
        b.exec()


if __name__ == "__main__":
    trial = "abd"
    subject = "Subject_1"
    mass = 62
    data_dir = f"data_final/{subject}"
    # model_path = f"{data_dir}/model_{subject}_scaled.bioMod"
    model_path = f"{data_dir}/model_{subject}.osim"
    initialize(model_path, data_dir, scaling=True, off_line=True, mass=mass)
