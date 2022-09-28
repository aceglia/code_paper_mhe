from biosiglive.io.save_data import read_data
import matplotlib.pyplot as plt
import glob
import numpy as np

subjects = ["subject_3"]  # , "subject_2"]
nb_none_conv = []
sol_freq = []
nb_total_iter = []
plot_std = []
plot_lat = []
process_lat = []
process_std = []
vicon_lat = []
vicon_std = []
for subject in subjects:
    # files = glob.glob(f"{subject}/plot_**")
    # # files = [f"{subject}/plot_delay_full_test_tronc_plot"]
    # for file in files:
    #     data_tmp = read_data(file)
    #     delay_tmp = data_tmp["plot_delay"]
    #     plot_lat.append(np.median(delay_tmp))
    #     plot_std.append(np.std(delay_tmp))

    files = glob.glob(f"{subject}/C3D/data_**")
    # files = [f"{subject}/data_streaming_20220127-1820_compressed"]
    for file in files:
        data_tmp = read_data(file)
        from biosiglive.io.save_data import add_data_to_pickle

        # add_data_to_pickle(data_tmp, f"{file}_compressed")
        nan = list(zip(*map(list, np.where(np.isnan(data_tmp["kalman"])))))
        delay_tmp = data_tmp["process_delay"]
        if nan:
            process_lat.append(np.median(delay_tmp[nan[0][1]]))
            process_std.append(np.std(delay_tmp[nan[0][1]]))
        else:
            process_lat.append(np.median(delay_tmp))
            process_std.append(np.std(delay_tmp))
        if "vicon_latency" in data_tmp:
            vicon_lat.append(np.median(data_tmp["vicon_latency"]))
            vicon_std.append(np.std(data_tmp["vicon_latency"]))
        else:
            vicon_lat.append([0])
            vicon_std.append([0])

    files = glob.glob(f"{subject}/C3D/results_w4/data**")

    for file in files:
        data_tmp = read_data(file)
        sol_freq.append(np.mean(data_tmp["sol_freq"][1:]))

plot_lat = np.round(np.mean(plot_lat), 2)
plot_std = np.round(np.mean(plot_std), 2)
total_plot_lat, total_plot_std = plot_lat * 1000, plot_std * 1000
total_process_lat, total_process_std = (
    np.round(np.mean(process_lat) * 1000, 2),
    np.round(np.mean(process_std) * 1000, 2),
)
total_vicon_lat, total_vicon_std = np.round(vicon_lat[0] * 1000, 2), np.round(vicon_std[0] * 1000, 2)

print(f"Plot latency : {total_plot_lat} ms +/- {total_plot_std} ms")
print("TCP/IP latency: 6.1 ms")
print(f"vicon latency : {total_vicon_lat} +/- {total_vicon_std} ms")
print(f"proc delay: {total_process_lat} ms +/- {total_process_std} ms")
n_round = 2


print(f"Solve frequency: {np.mean([i for i in sol_freq])} Hz +/- {np.std(sol_freq)}")
print(f"Solve duration: {1/np.mean(sol_freq)*1000} ms +/- {np.std([1/i for i in sol_freq])*1000}")

# print(
#     f"\hline computer 1 & Nexus & ${total_vicon_lat}$ & ${total_vicon_std}$\ \ \n "
#     f"& Processing & ${total_process_lat}$ & ${total_process_std}$\ \ \n"
#     f"& TCP/IP & ${6.1}$ & ${1.2}$\ \ \n"
#     f"\hline"
#     f"computer 2 & Estimator & ${np.round(1/np.mean(sol_freq)*1000,n_round)}$ & ${np.std([1/i for i in sol_freq]) *1000}$\ \ \n "
#     f"& Vizualisation & ${total_plot_lat}$ & ${total_plot_std}$\ \ \n"
#     f"\hline"
#     f"& Total & ${total_process_lat+total_vicon_lat+total_plot_lat+1/np.mean(sol_freq)*1000+6.1}$ &"
#     f" ${total_process_std+total_vicon_std+total_plot_std + np.std([1/i for i in sol_freq]) *1000+1.2 }$\ \ \n"
# )
