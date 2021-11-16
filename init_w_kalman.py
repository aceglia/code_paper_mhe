import biorbd
import bioviz
from biosiglive.client import Client
from biosiglive.server import Server
import numpy as np
from time import sleep

server_ip = "192.168.1.211"
server_port = 50000
n_marks = 15
model_path = "models/wu_model.bioMod"
for i in range(5):
        client = Client(server_ip, server_port, "TCP")
        markers_tmp = client.get_data(
            ["markers"], nb_frame_of_interest=100, read_frequency=100, nb_of_data_to_export=10, get_names=True
        )  # , get_kalman=False)
        sleep((1 / 100) * 10)
        if i == 0:
            mark_0 = markers_tmp["markers"]
            markers = np.array(mark_0).reshape((3, n_marks, 10))
        #
        else:
            mark_tmp = markers_tmp["markers"]
            mark_tmp = np.array(mark_tmp).reshape((3, n_marks, 10))
            markers = np.append(markers, mark_tmp, axis=2)

bmodel = biorbd.Model(model_path)
q_recons, _ = Server.kalman_func(markers, model=bmodel)
b = bioviz.Viz(model_path=model_path)
b.load_movement(q_recons)
b.exec()

print(np.mean(q_recons[:6, :], axis=1))
