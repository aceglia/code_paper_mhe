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
from biosiglive.io.save_data import read_data
from osim_to_biomod import Converter


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


def initialize(
    osim_model: str = None,
    biomod_model: str = None,
    data_dir: str = "data",
    scaling: bool = False,
    mass: float = None,
    trial: str = None,
):
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
    mat = read_data(f"{data_dir}/{trial}")
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

    if scaling:
        model_path = osim_model
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
        model_path = biomod_model
        bmodel = biorbd.Model(model_path)
        q_recons, _ = kalman_func(markers, model=bmodel, use_kalman=True)
        q_mean = q_recons.mean(axis=1)
        print(q_mean[3], q_mean[4], q_mean[5], " xyz ", q_mean[0], q_mean[1], q_mean[2])
        b = bioviz.Viz(model_path=model_path)
        b.load_movement(q_recons)  # Q from kalman array(nq, nframes)
        b.load_experimental_markers(markers)  # experimental markers array(3, nmarkers, nframes)
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
    converter = Converter(
        out_path, in_path, ignore_clamped_dof_tag=True, ignore_muscle_applied_tag=True, print_warnings=False
    )
    converter.convert_file()
    if viz:
        b = bioviz.Viz(model_path=out_path)
        b.exec()


if __name__ == "__main__":
    trial = "data_abd_sans_poid"
    bio_model = "data/wu_scaled.bioMod"
    mass = 72
    mass_scaling = mass * 0.578 + mass * 0.050
    data_dir = f"data"
    osim_model = f"{data_dir}/wu.osim"
    initialize(
        osim_model=osim_model, biomod_model=bio_model, data_dir=data_dir, scaling=False, mass=mass_scaling, trial=trial
    )
