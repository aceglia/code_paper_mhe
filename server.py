from biosiglive.server import Server

if __name__ == '__main__':
    server_ip = "127.0.0.1"
    server_port = 50000
    read_freq = 100
    n_electrode = 10
    subject = "Remi"
    data_dir = "/home/amedeo/Documents/programmation/data_article/Remi/"
    # data_dir = f"data/test_09_12_21/{subject}/"
    motion = 'abd_co'
    offline_file_path = data_dir + "test_" + motion + '_for_server.mat'

    # Run streaming data
    muscles_idx = (0, n_electrode - 1)
    server = Server(
            IP=server_ip,
            server_ports=server_port,
            device="vicon",
            type="TCP",
            muscle_range=muscles_idx,
            acquisition_rate=read_freq,
            model_path=data_dir + f"Wu_Shoulder_Model_mod_wt_wrapp_{subject}.bioMod",
            recons_kalman=True,
            output_dir=data_dir,
            output_file='test_' + motion,
            offline_file_path=offline_file_path
        )

    server.run(
        stream_emg=True,
        stream_markers=True,
        stream_imu=False,
        optim=False,
        plot_emg=False,
        norm_emg=False,
        test_with_connection=False,
    )