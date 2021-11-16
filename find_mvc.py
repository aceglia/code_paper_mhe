from biosiglive.live_mvc import ComputeMvc

if __name__ == "__main__":
    # number of EMG electrode
    n_electrode = 2

    # set file and directory to save
    file_name = "MVC.mat"
    file_dir = "MVC_18_11_2021"
    device_host = "192.168.1.211"
    muscle_names = ["tri_long", "bic_long"]
    # Run MVC
    muscles_idx = (0, n_electrode - 1)
    MVC = ComputeMvc(
        stream_mode="server_data",
        server_ip="localhost",
        server_port=50000,
        range_muscles=muscles_idx,
        output_dir=file_dir,
        device_host=device_host,
        output_file=file_name,
        test_with_connection=False,
        muscle_names=muscle_names,
    )
    list_mvc = MVC.run()
    print(list_mvc)
