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
from models.scaling import C3DtoTRC
import csv
from biosiglive.client import Client
from biosiglive.server import Server
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


def EKF(model_path, scaling=False):
    server_ip = "192.168.1.211"
    server_port = 50000
    n_marks = 16
    for i in range(5):
        client = Client(server_ip, server_port, "TCP")
        markers_tmp = client.get_data(
            ["markers"], nb_frame_of_interest=100, read_frequency=100, nb_of_data_to_export=10, get_names=True
        )  # , get_kalman=False)
        sleep((1 / 100) * 10)
        if i == 0:
            mark_0 = markers_tmp["markers"]
            marker_names = markers_tmp["marker_names"]
            markers = np.array(mark_0).reshape((3, n_marks, 10))
        #
        else:
            mark_tmp = markers_tmp["markers"]
            mark_tmp = np.array(mark_tmp).reshape((3, n_marks, 10))
            markers = np.append(markers, mark_tmp, axis=2)

    # use_col = ["MAN", "XYP","C7","T10","CLAV_SC","CLAV_AC","SCAP_IA","SCAP_AA","SCAP_AC",
    #  "EPI_lat","EPI_med","ELB","DELT","ARM",
    #  "ULNA","RADIUS"]
    # from pyomeca import Markers
    # import scipy.io as sio
    # markers = sio.loadmat("data/data_30_11_21/RT_test/test_abd.mat")["markers"][:3, :, :]
    # markers = Markers.from_c3d("/home/amedeo/Documents/programmation/code_paper_mhe/data/data_09_2021/flex_co.c3d")#, usecols=use_col)

    if scaling:
        # ---------- model scaling ------------ #
        marker_names = ['STER', 'XIPH', 'C7', 'T10', 'CLAV_SC', 'CLAV_AC',
                        'SCAP_IA', 'SCAP_AA', 'Acrom', 'EPICl', 'EPICm', 'LARM_elb', 'DELT', 'ARMl', 'STYLu',
                        'STYLr']
        from pathlib import Path
        osim_model_path = "models/Wu_Shoulder_Model_mod_wt_wrapp.osim"
        model_output = "models/" + Path(osim_model_path).stem + '_remi_scaled.osim'
        scaling_tool = "scale_tool.xml"
        trc_file = 'data/data_30_11_21/RT_test/test_abd.trc'
        C3DtoTRC.WriteTrcFromMarkersData(trc_file,
                                         markers=markers[:3, :, :],
                                         marker_names=marker_names,
                                         data_rate=100,
                                         cam_rate=100,
                                         n_frames=markers.shape[2],
                                         units="m").write()

        import opensim as osim
        pyosim.Scale(model_input=osim_model_path,
                     model_output=model_output,
                     xml_input=scaling_tool,
                     xml_output='scaling_tool_output.xml',
                     static_path=trc_file
                     )

        convert_model(in_path="models/" + Path(model_output).stem + "_markers.osim",
                      out_path="models/" + Path(model_output).stem + '.bioMod', viz=False)

    else:
        # from pyomeca import Markers
        bmodel = biorbd.Model(model_path)
        # marker_names = []
        # for i in range(len(bmodel.markerNames())):
        #     marker_names.append(bmodel.markerNames()[i].to_string())
        # markers_full = Markers.from_c3d(f"data/test_18_11_21/gregoire/test_1/flex.c3d", usecols=marker_names)
        # marker_rate = int(markers_full.rate)
        # marker_exp = markers_full[:, :, :].data * 1e-3
        q_recons, _ = Server.kalman_func(markers, model=bmodel)
        b = bioviz.Viz(model_path=model_path)
        b.load_movement(q_recons)
        b.load_experimental_markers(markers)
        b.exec()


def convert_model(in_path, out_path, viz=None):
    #  convert_model
    from OsimToBiomod import Converter

    converter = Converter(out_path, in_path)
    converter.main()
    if viz:
        b = bioviz.Viz(model_path=out_path)
        b.exec()


if __name__ == '__main__':
    model_in = "data/data_30_11_21/Wu_Shoulder_Model_mod_wt_wrapp_remi_scaled.osim"
    model_out = "data/data_30_11_21/Wu_Shoulder_Model_mod_wt_wrapp_remi_scaled.bioMod"
    # convert_model(in_path=model_in, out_path=model_out, viz=True)
    model_path = "data/test_09_12_21/Mathis/Wu_Shoulder_Model_mod_wt_wrapp_Mathis.bioMod"
    EKF(model_path, scaling=False)

