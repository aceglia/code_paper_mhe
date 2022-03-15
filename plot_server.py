import socket
import struct
import json
import numpy as np
import biorbd
from biosiglive.data_plot import init_plot_force, update_plot_force

Buff_size = 100000


def recv_all(connection, buff_size):
    msg_len = connection.recv(4)
    msg_len = struct.unpack(">I", msg_len)[0]
    data = []
    l = 0
    while l < msg_len:
        chunk = connection.recv(buff_size)
        l += len(chunk)
        data.append(chunk)
    data = b"".join(data)
    data = json.loads(data)
    return data


if __name__ == "__main__":
    # Open server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    plot_server_ip = "127.0.0.1"
    plot_server_port = 50001
    server.bind((plot_server_ip, plot_server_port))
    server.listen(10)
    print("Server is listening, waiting for client connection.")
    count = 0
    p_force, app_force, win_force = 0, 0, 0
    data_to_show = []

    # Waiting for data
    b = None
    while True:
        connection, ad = server.accept()
        message = recv_all(connection, Buff_size)
        if count == 0:
            conf_problem = message
            data_to_show = message["data_to_show"]
            model_path = message["model_path"]
            model = biorbd.Model(model_path)
            nbMT = model.nbMuscles()
            for data in data_to_show:
                if data == "force":
                    p_force, win_force, app_force = init_plot_force(nbMT)
                if data == "q":
                    import bioviz

                    b = bioviz.Viz(
                        model_path=model_path,
                        show_global_center_of_mass=False,
                        show_markers=True,
                        show_floor=False,
                        show_gravity_vector=False,
                        show_muscles=False,
                        show_segments_center_of_mass=False,
                        show_local_ref_frame=False,
                        show_global_ref_frame=False,
                    )

            model = biorbd.Model(model_path)
            muscle_names = []
            for i in range(model.nbMuscles()):
                muscle_names.append(model.muscleNames()[i].to_string())
            count += 1

        else:
            force_to_plot = np.array(message["force_est"])
            # muscle_names = message["muscle_names"]
            q_est = np.array(message["q_est"])
            if "force" in data_to_show:
                update_plot_force(force_to_plot, p_force, app_force, ratio=1, muscle_names=muscle_names)  # , box_force)
            if "q" in data_to_show:
                b.set_q(q_est[:, -1])
