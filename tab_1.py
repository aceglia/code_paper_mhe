from biosiglive.file_io.save_and_load import load
import matplotlib.pyplot as plt
import glob
import numpy as np

nb_none_conv = []
sol_freq = []
nb_total_iter = []
plot_std = []
plot_lat = []
process_lat = []
process_std = []
vicon_lat = []
vicon_std = []
files = glob.glob(f"data/plot_**")
for file in files:
    data_tmp = load(file)
    delay_tmp = data_tmp["plot_delay"]
    plot_lat.append(np.median(delay_tmp))
    plot_std.append(np.std(delay_tmp))

files = glob.glob(f"data/data_**")
for file in files:
    try:
        data_tmp = load(file)
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
    except:
        pass

files = glob.glob(f"results/results_all_trials_w6_freq_calc")
for file in files:
    data_tmp = load(file)
    for key in data_tmp.keys():
        sol_freq.append(np.mean(data_tmp[key][0]["0.08"]["75"]["sol_freq"][1:]))

plot_lat = np.round(np.mean(plot_lat), 5)
plot_std = np.round(np.mean(plot_std), 5)
total_plot_lat, total_plot_std = plot_lat * 1000, plot_std * 1000
total_process_lat, total_process_std = (
    np.round(np.mean(process_lat) * 1000, 5),
    np.round(np.mean(process_std) * 1000, 5),
)
total_vicon_lat, total_vicon_std = np.round(vicon_lat[0] * 1000, 5), np.round(vicon_std[0] * 1000, 5)

print(f"Plot latency : {total_plot_lat} ms +/- {total_plot_std} ms")
print("TCP/IP latency: 6.1 ms")
print(f"vicon latency : {total_vicon_lat} +/- {total_vicon_std} ms")
print(f"proc delay: {total_process_lat} ms +/- {total_process_std} ms")
n_round = 2


print(f"Solve frequency: {np.mean([i for i in sol_freq])} Hz +/- {np.std(sol_freq)}")
print(f"Solve duration: {1/np.mean(sol_freq)*1000} ms +/- {np.std([1/i for i in sol_freq])*1000}")
print(f"frame_saved: {1/np.mean(sol_freq)*1000*0.75} ms +/- {np.std([1/i for i in sol_freq])*1000*0.75}")

print(
    f"\hline computer 1 & Nexus & ${total_vicon_lat}$ & ${total_vicon_std}$\ \ \n "
    f"& Processing & ${total_process_lat}$ & ${total_process_std}$\ \ \n"
    f"& TCP/IP & ${6.1}$ & ${1.2}$\ \ \n"
    f"\hline"
    f"computer 2 & Estimator & ${np.round(1/np.mean(sol_freq)*1000,n_round)}$ & ${np.std([1/i for i in sol_freq]) *1000}$\ \ \n "
    f"& Vizualisation & ${total_plot_lat}$ & ${total_plot_std}$\ \ \n"
    f"\hline"
    f"& Total & ${total_process_lat+total_vicon_lat+total_plot_lat+1/np.mean(sol_freq)*1000+6.1+1/np.mean(sol_freq)*1000*0.75}$ &"
    f" ${total_process_std+total_vicon_std+total_plot_std + np.std([1/i for i in sol_freq]) *1000+1.2 +np.std([1/i for i in sol_freq])*1000*0.75}$\ \ \n"
)
