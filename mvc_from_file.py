import os

from biosiglive.processing.data_processing import OfflineProcessing
from biosiglive.io.save_data import add_data_to_pickle
from pyomeca import Analogs, Markers
import glob
import numpy as np

channel_names = ["pec.IM EMG1", "delt_ant.IM EMG2"
, "delt_med.IM EMG3",
"delt_post.IM EMG4",
"bic.IM EMG5",
"tri.IM EMG6",
"trap_sup.IM EMG7",
"trap_med.IM EMG8",
"trap_inf.IM EMG9",
"lat.IM EMG10"]
# emg_proc = []
# nb_muscles = 10
# data_dir = "data_final_new/subject_3/C3D"
# c3d_files = glob.glob(data_dir + "/MVC**")
# for file in c3d_files:
#     emg_tmp = Analogs.from_c3d(file, usecols=channel_names)
#     if len(emg_proc) == 0:
#         emg_proc = OfflineProcessing().process_emg(emg_tmp.values, 2000, ma=True)
#     else:
#         emg_proc = np.append(emg_proc, OfflineProcessing().process_emg(emg_tmp.values, 2000, ma=True), axis=1)
#
# mvc_windows = 200
# mvc_list_max = np.ndarray((nb_muscles, mvc_windows))
# save = True
# output_file = "mvc.mat"
# mvc = OfflineProcessing.compute_mvc(nb_muscles,
#                                     emg_proc,
#                                     mvc_windows,
#                                     mvc_list_max,
#                                     None,
#                                     output_file,
#                                     save)
# print(mvc)
markers_name = ['STER', 'XIPH', 'C7', 'T10', 'CLAV_SC', 'CLAV_AC', 'SCAP_IA', 'Acrom',
       'SCAP_AA', 'EPICl', 'EPICm', 'DELT', 'ARMl', 'LARM_elb', 'STYLu',
       'STYLr']
data_dir = "data_final_new/subject_3/C3D"
c3d_files = glob.glob(data_dir + "/**")
import matplotlib.pyplot as plt
for file in c3d_files:
    if "MVC" not in file:
        print(file)
        markers = Markers.from_c3d(file, usecols=markers_name)
        emg = Analogs.from_c3d(file, usecols=channel_names)
        emg_processed = (
            emg.meca.band_pass(order=2, cutoff=[10, 425])
            .meca.center()
            .meca.abs()
            .meca.low_pass(order=4, cutoff=5, freq=emg.rate)
            # .meca.normalize()
        )

        emg_processed.plot(col="channel", col_wrap=3)
        nan_idx = list(zip(*map(list, np.where(np.isnan(markers.values)))))
        for idx in nan_idx:
            markers.values[idx] = 0
        plt.plot(markers.values[1, -1, :])
        plt.show()
        n_frame = int(input("n_frame"))
        data = {"raw_emg": emg.values[:, n_frame * 20:], "markers": markers.values[:3, :, n_frame:]}
        if os.path.isfile(file[:-4]):
            os.remove(file[:-4])
        add_data_to_pickle(data, file[:-4])
