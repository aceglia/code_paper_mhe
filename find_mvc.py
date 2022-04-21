"""
Script to find MVCs in real time using the mvc tool provided in biosiglive package.
"""

from biosiglive.live_mvc import ComputeMvc

if __name__ == "__main__":
    # number of EMG electrode
    n_electrode = 10

    # set file and directory to save
    file_name = "MVC.mat"
    file_dir = "/home/amedeo/Documents/programmation/code_paper_mhe/data/test_27_01_22/Clara/"
    server_ip = "192.168.1.211"
    device_host = "192.168.1.211"
    muscle_names = ["pec", "DA", "DM", "DP", "bic", "tri", "trap_sup", "trap_med", "trap_inf", "lat"]
    # Run MVC
    muscles_idx = (0, n_electrode - 1)
    MVC = ComputeMvc(
        stream_mode="server_data",
        server_ip=server_ip,
        server_port=50000,
        range_muscles=muscles_idx,
        output_dir=file_dir,
        device_host=device_host,
        output_file=file_name,
        test_with_connection=True,
        muscle_names=muscle_names,
    )
    list_mvc = MVC.run()
    print(list_mvc)
