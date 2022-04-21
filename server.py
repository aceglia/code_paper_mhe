"""
Script to run the server on the acquisition computer using biosiglive package.
"""

from biosiglive.server import Server

if __name__ == "__main__":
    server_ip = "192.168.1.211"
    server_port = 50000
    read_freq = 100
    n_electrode = 10
    subject = "Subject_1"
    data_dir = f"data_final/{subject}/"
    motion = "abd_co"

    # Run streaming data
    muscles_idx = (0, n_electrode - 1)
    server = Server(
        IP=server_ip,
        server_ports=server_port,
        device="vicon",
        type="TCP",
        muscle_range=muscles_idx,
        acquisition_rate=read_freq,
        model_path=data_dir + f"model_{subject}_scaled.bioMod",
        recons_kalman=True,
        output_dir=data_dir,
        output_file=motion,
    )

    server.run(
        stream_emg=True,
        stream_markers=True,
        stream_imu=False,
        optim=True,
        plot_emg=False,
        norm_emg=False,
        test_with_connection=True,
    )
