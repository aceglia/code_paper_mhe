from biosiglive.server import Server

if __name__ == '__main__':
    server_ip = "127.0.0.1"
    server_port = 50000
    read_freq = 100
    n_electrode = 10

    # load MVC data from previous trials.
    # file_name = "MVC_xxxx.mat"
    # file_dir = "MVC_01_08_2021"
    # list_mvc = sio.loadmat(f"{file_dir}/{file_name}")["MVC_list_max"]
    # list_mvc = list_mvc[:, :n_electrode].T
    # Set file to save data
    # output_file = "stream_data_xxx"
    # output_dir = "tests"
    data_dir = "data/test_09_12_21/Jules/"
    motion = 'abd'
    offline_file_path = data_dir + "test_" + motion + '.mat'
    # Run streaming data
    muscles_idx = (0, n_electrode - 1)
    server = Server(
            IP=server_ip,
            server_ports=server_port,
            device="vicon",
            type="TCP",
            muscle_range=muscles_idx,
            # device_host_ip=device_ip,
            acquisition_rate=read_freq,
            model_path=data_dir + "Wu_Shoulder_Model_mod_wt_wrapp_Jules.bioMod",
            recons_kalman=True,
            # output_dir=output_dir,
            # output_file=output_file,
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
        # mvc_list=MVC_list
    )