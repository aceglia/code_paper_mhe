from biosiglive.client import Client
import multiprocessing as mp
import biorbd_casadi as biorbd
from mhe.ocp import *
import casadi
from mhe.utils import *
import json
import struct
import socket

# TODO add class for configurate problem instead of dic
# class ConfPlot:


class MuscleForceEstimator:
    def __init__(self, *args):
        conf = check_and_adjust_dim(*args)
        self.model_path = conf["model_path"]  # arm26_6mark_EKF.bioMod"
        biorbd_model = biorbd.Model(self.model_path)
        self.use_torque = False
        self.use_excitation = False
        self.save_results = True
        self.track_emg = False
        self.use_N_elec = False
        self.data_to_show = [""]
        self.kin_data_to_track = None
        self.init_w_kalman = False
        self.is_mhe = True
        self.test_offline = False
        self.offline_file = None

        # Variables of the problem
        self.exp_freq = 20
        self.ns_mhe = 0
        self.mhe_time = 0.1
        self.markers_rate = 100
        self.emg_rate = 2000
        self.get_names = False
        self.get_kalman = True
        self.muscle_track_idx = []

        # define some variables
        self.var, self.server_ip, self.server_port, self.data_to_show = {}, [], [], []

        # multiprocess stuffs
        manager = mp.Manager()
        self.data_count = mp.Value('i', 0)
        self.plot_queue = manager.Queue()
        self.data_queue = manager.Queue()
        self.data_event = mp.Event()
        self.process = mp.Process
        self.plot_event = mp.Event()

        self.p_q, self.win_q, self.app_q, self.box_q = [], [], [], []
        self.p_force, self.win_force, self.app_force, self.box_force = [], [], [], []
        self.plot_force_ratio, self.plot_q_ratio = 0, 0
        self.print_lvl = 1
        self.plot_q_freq, self.plot_force_freq = self.exp_freq, 10
        self.force_to_plot, self.q_to_plot = [], []
        self.count_p_f, self.count_p_q = [], []
        self.mvc_list = []
        self.interpol_factor = 1
        self.weights = {}
        self.result_dir = None
        self.ns_full = 0

        for key in conf.keys():
            self.__dict__[key] = conf[key]
        self.T_mhe = self.mhe_time
        self.ns_mhe = int(self.T_mhe * self.markers_rate * self.interpol_factor)
        self.slide_size = int((1 / self.exp_freq) / (1 / self.markers_rate) * self.interpol_factor)
        self.nbQ, self.nbMT = biorbd_model.nbQ(), biorbd_model.nbMuscles()
        self.nbGT = biorbd_model.nbGeneralizedTorque() if self.use_torque else 0
        self.current_time = strftime("%Y%m%d-%H%M")
        self.data_to_get = []
        self.data_to_get.append("markers")
        self.data_to_get.append("emg")
        # self.muscle_track_idx = [13, 15, 21, 10, 1, 2, 27, 14, 25, 26, 23, 24, 28, 29, 30]

        self.markers_ratio = 1  # int(self.markers_rate / self.exp_freq)
        self.EMG_ratio = 1  # int(self.emg_rate / self.exp_freq)
        self.rt_ratio = self.markers_ratio
        self.muscle_names = []
        for i in range(biorbd_model.nbMuscles()):
            self.muscle_names.append(biorbd_model.muscleNames()[i].to_string())
        self.dof_names = []
        for i in range(biorbd_model.nbQ()):
            self.dof_names.append(biorbd_model.nameDof()[i].to_string())

    def prepare_problem(self):
        biorbd_model = biorbd.Model(self.model_path)

        self.data_to_get = []
        # if self.kin_data_to_track == "markers":
        self.data_to_get.append("markers")
        # elif self.kin_data_to_track == "q":
        #     self.data_to_get.append("q")
        # if self.track_emg:
        self.data_to_get.append("emg")
        from scipy import interpolate
        if self.test_offline is True:
            x_ref, markers_target, muscles_target = get_reference_data(offline_path)
            muscles_target = muscles_target[:, :]
            # x_ref = x_ref[:, :]
            x_ref = x_ref[6:, :]
        else:
            nb_of_data = self.ns_mhe + 1  # if t == 0 else 2
            vicon_client = Client(self.server_ip, self.server_port, type="TCP")
            data = vicon_client.get_data(
                self.data_to_get,
                read_frequency=self.markers_rate,
                nb_of_data_to_export=nb_of_data,
                nb_frame_of_interest=self.ns_mhe,
                get_names=self.get_names,
                get_kalman=self.get_kalman,
                norm_emg=False,
                # mvc_list=self.mvc_list
            )
            x_ref = np.array(data["kalman"])[6:, :]
            markers_target = np.array(data["markers"])
            muscles_target = np.array(data["emg"])
        window_len = self.ns_mhe if self.is_mhe else self.ns_full
        window_duration = self.T_mhe if self.is_mhe else window_len / self.markers_rate
        # interpolate target
        if self.interpol_factor != 1:
            # x_ref
            x = np.linspace(0, x_ref.shape[1] / 100, x_ref.shape[1])
            f_x = interpolate.interp1d(x, x_ref)
            x_new = np.linspace(0, x_ref.shape[1] / 100, int(x_ref.shape[1] * self.interpol_factor))
            self.x_ref = f_x(x_new)

            # markers_ref
            self.markers_target = np.zeros(
                (3, biorbd_model.nbMarkers(), int(markers_target.shape[2] * self.interpol_factor)))
            for i in range(3):
                x = np.linspace(0, markers_target.shape[2] / 100, markers_target.shape[2])
                f_mark = interpolate.interp1d(x, markers_target[i, :, :])
                x_new = np.linspace(0, markers_target.shape[2] / 100,
                                    int(markers_target.shape[2] * self.interpol_factor))
                self.markers_target[i, :, :] = f_mark(x_new)

            # muscle_target
            x = np.linspace(0, muscles_target.shape[1] / 100, muscles_target.shape[1])
            f_mus = interpolate.interp1d(x, muscles_target)
            x_new = np.linspace(0, muscles_target.shape[1] / 100,
                                int(muscles_target.shape[1] * self.interpol_factor))
            muscles_target = f_mus(x_new)
        else:
            self.markers_target = markers_target
            self.x_ref = x_ref

        self.muscles_target = np.zeros(
            (len(self.muscle_track_idx), int(muscles_target.shape[1])))

        self.muscles_target[[0, 1, 2], :] = muscles_target[0, :]
        self.muscles_target[[3], :] = muscles_target[1, :]
        self.muscles_target[4, :] = muscles_target[2, :]
        self.muscles_target[5, :] = muscles_target[3, :]
        self.muscles_target[[6, 7], :] = muscles_target[4, :]
        self.muscles_target[[8, 9, 10], :] = muscles_target[5, :]
        self.muscles_target[[11], :] = muscles_target[6, :]
        self.muscles_target[[12], :] = muscles_target[7, :]
        self.muscles_target[[13], :] = muscles_target[9, :]
        # self.muscles_target[[14], :] = muscles_target[9, :]
        self.muscles_target = self.muscles_target / np.repeat(
            mvc_list, muscles_target.shape[1]).reshape(len(mvc_list), muscles_target.shape[1])

        # self.x_ref = np.zeros((biorbd_model.nbQ(), self.ns_mhe + 1))
        # casadi funct:
        # self.markers_target = np.ndarray((3, biorbd_model.nbMarkers(), 1))
        # q_sym = casadi.MX.sym("Q", biorbd_model.nbQ(), 1)
        # markers = biorbd.to_casadi_func("Markers", biorbd_model.markers, q_sym)
        # self.markers_target[:, :, 0] = np.array(markers(self.x_ref[:biorbd_model.nbQ(), -1]))
        # self.markers_target = np.repeat(self.markers_target, self.ns_mhe + 1, axis=2)

        self.muscles_target = self.muscles_target[:, :window_len]
        self.kin_target = self.markers_target[:, :, :window_len + 1] if self.kin_data_to_track == "markers" else self.x_ref[:self.nbQ, :window_len+1]

        for i in range(biorbd_model.nbMuscles()):
            self.muscle_names.append(biorbd_model.muscleNames()[i].to_string())
        self.dof_names = []
        for i in range(biorbd_model.nbQ()):
            self.dof_names.append(biorbd_model.nameDof()[i].to_string())
        objectives = define_objective(
            weights=self.weights,
            use_excitation=self.use_excitation,
            use_torque=self.use_torque,
            track_emg=self.track_emg,
            muscles_target=self.muscles_target,
            kin_target=self.kin_target,
            biorbd_model=biorbd_model,
            kin_data_to_track=self.kin_data_to_track,
            muscle_track_idx=self.muscle_track_idx
        )

        self.mhe, self.solver = prepare_problem(
            biorbd_model,
            objectives,
            window_len=window_len,
            window_duration=window_duration,
            x0=self.x_ref,
            use_torque=self.use_torque,
            use_excitation=self.use_excitation,
            nb_threads=8,
            is_mhe=self.is_mhe
        )
        if not self.is_mhe:
            self.ocp = self.mhe
        self.get_force = force_func(biorbd_model, use_excitation=self.use_excitation)
        self.force_est = np.ndarray((biorbd_model.nbMuscles(), 1))
    # @staticmethod

    def get_data(self, multi=False):
        # self.plot_event.wait()
        data_to_get = self.data_to_get
        # stream_frequency = 2 * self.exp_freq
        # if multi:
        #     while True:
        #         # tic = time()
        #         try:
        #             self.data_queue.get_nowait()
        #         except:
        #             pass
        #             # self.data_count.value = 0
        #         nb_of_data = self.ns_mhe + 1  # if t == 0 else 2
        #         vicon_client = Client(self.server_ip, self.server_port, type="TCP")
        #         data = vicon_client.get_data(
        #             data_to_get,
        #             read_frequency=self.exp_freq,
        #             nb_of_data_to_export=nb_of_data,
        #             nb_frame_of_interest=self.ns_mhe,
        #             get_names=self.get_names,
        #             get_kalman=self.get_kalman,
        #             norm_emg=True,
        #             mvc_list=self.mvc_list
        #         )
        #         self.data_queue.put_nowait(data)
        # else:
        nb_of_data = self.ns_mhe + 1  # if t == 0 else 2
        vicon_client = Client(self.server_ip, self.server_port, type="TCP")
        data = vicon_client.get_data(
            data_to_get,
            read_frequency=self.exp_freq,
            nb_of_data_to_export=nb_of_data,
            nb_frame_of_interest=self.ns_mhe,
            get_names=self.get_names,
            get_kalman=self.get_kalman,
            norm_emg=False,
            mvc_list=self.mvc_list
        )
        return data

    def run(self,
            var,
            server_ip,
            server_port,
            data_to_show=None,
            test_offline=False,
            offline_file=None,
            data_process=False
            ):
        self.var = var
        self.server_ip = server_ip
        self.server_port = server_port
        self.data_to_show = data_to_show
        self.test_offline = test_offline
        self.offline_file = offline_file
        if self.test_offline and not self.offline_file:
            raise RuntimeError("Please provide a data file to run offline program")
        if not self.is_mhe and not self.test_offline:
            raise RuntimeError("Online optimisation only available using MHE implementation")

        proc_plot, proc_get_data = [], []
        self.data_process = data_process
        t = 0
        # if test_offline is not True and data_process:
        #     proc_get_data = self.process(name="data", target=MuscleForceEstimator.get_data, args=(self, t))
        #     proc_get_data.start()

        if self.data_to_show:
            proc_plot = self.process(name="plot", target=MuscleForceEstimator.run_plot, args=(self,))
            # proc_plot = self.process(name="plot", target=MuscleForceEstimator.send_plot_data, args=(self,))
            proc_plot.start()

        proc_mhe = self.process(name="mhe", target=MuscleForceEstimator.run_mhe, args=(self, var, server_ip, server_port, data_to_show))
        proc_mhe.start()
        # MuscleForceEstimator.run_mhe(self, var, server_ip, server_port, data_to_show)

        if self.data_to_show:
            proc_plot.join()

        proc_mhe.join()
        # if test_offline is not True and data_process:
        #     proc_get_data.join()

    def send_plot_data(self):
        server_address = '127.0.0.1'
        port = 50001
        # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client.connect((server_address, port))
        dict_to_send = {"model_path": self.model_path, "data_to_show": self.data_to_show}
        encoded_data = json.dumps(dict_to_send).encode()
        encoded_data = struct.pack('>I', len(encoded_data)) + encoded_data
        # client.sendall(encoded_data)
        data = []
        self.plot_event.set()
        while True:
            try:
                data = self.plot_queue.get_nowait()
                is_working = True
            except:
                is_working = False

            if is_working:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # client.connect((server_address, port))
                encoded_data = json.dumps(data).encode()
                encoded_data = struct.pack('>I', len(encoded_data)) + encoded_data
                # client.sendall(encoded_data)

    def run_plot(self):
        for data_to_show in self.data_to_show:
            if data_to_show == "force":
                # p_force, win_force, app_force, box_force = init_plot_force(self.nbMT)
                self.p_force, self.win_force, self.app_force = init_plot_force(self.nbMT)

            if data_to_show == "q":
                # self.p_q, self.win_q, self.app_q, self.box_q = init_plot_q(self.nbQ, self.dof_names)
                import bioviz
                self.b = bioviz.Viz(model_path=self.model_path,
                                    show_global_center_of_mass=False,
                                    show_markers=True,
                                    show_floor=False,
                                    show_gravity_vector=False,
                                    show_muscles=False,
                                    show_segments_center_of_mass=False,
                                    show_local_ref_frame=False,
                                    show_global_ref_frame=False
                                    )
        self.q_to_plot = np.zeros((self.nbQ, self.plot_q_ratio))
        self.plot_q_ratio = int(self.exp_freq / self.plot_q_freq)
        self.plot_force_ratio = int(self.exp_freq / self.plot_force_freq)
        self.force_to_plot = np.zeros((self.nbMT, self.plot_force_ratio))
        self.count_p_f, self.count_p_q = self.plot_force_ratio, self.plot_q_ratio
        self.plot_event.set()
        while True:
            try:
                data = self.plot_queue.get_nowait()
                is_working = True
            except:
                is_working = False
            if is_working:
                # from mhe.utils import update_plot
                update_plot(self, data["t"], data["force_est"], data["q_est"])

    def run_mhe(self, var, server_ip, server_port, data_to_show):
        self.prepare_problem()
        if data_to_show:
            self.plot_event.wait()
        import sys
        for key in var.keys():
            if key in self.__dict__:
                self.__setattr__(key, var[key])
            else:
                raise RuntimeError(f'{key} is not a variable of the class')

        if self.test_offline:
            x_ref, markers_ref, muscles_ref = get_reference_data(self.offline_file)
            offline_data = [x_ref[6:, :], markers_ref, muscles_ref[:-1, :]]
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(muscles_ref.T)
            # plt.show()
        else:
            offline_data = None
        self.model = biorbd.Model(self.model_path)
        initial_time = time()
        # muscle_track_idx = [0, 1, 2, 3, 4, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26]
        if self.is_mhe:
            final_sol = self.mhe.solve(
                lambda mhe, i, sol: update_mhe(mhe, i, sol, self,
                                               initial_time=initial_time,
                                               muscle_track_idx=self.muscle_track_idx,
                                               offline_data=offline_data),
                export_options={'frame_to_export': self.ns_mhe - 1},
                solver=self.solver
            )
            # final_sol.animate()
            final_sol.graphs()
        else:
            final_sol = self.ocp.solve(solver=self.solver)
            print(final_sol.status)
            # final_sol.animate()
            # final_sol.graphs()
            if self.save_results:
                data_to_save = {
                    "time": time() - initial_time,
                    "X_est": final_sol.states["all"],
                    "U_est": final_sol.controls["muscles"],
                    "kalman": self.x_ref[:, :self.ns_full],
                    # "f_est": force_est,
                    "init_w_kalman": self.init_w_kalman,
                    "none_conv_iter": final_sol.status,
                    "Nmhe": self.ns_full,
                    "muscles_target": self.muscles_target[:, :self.ns_full] if self.track_emg else
                    np.zerros((len(self.muscle_track_idx), self.ns_full))
                }
                if self.kin_data_to_track == 'q':
                    kin_target = self.x_ref[:, :self.ns_full]
                else:
                    kin_target = self.markers_target[:, :self.ns_full]

                data_to_save["kin_target"] = kin_target
                data_to_save["sol_freq"] = 1 / data_to_save["time"]
                data_to_save["exp_freq"] = self.exp_freq
                data_to_save["sleep_time"] = (1 / self.exp_freq) - data_to_save["time"]
                save_results(final_sol,
                             data_to_save,
                             self.current_time,
                             self.kin_data_to_track,
                             self.track_emg,
                             self.use_torque,
                             self.use_excitation,
                             self.result_dir,
                             is_mhe=self.is_mhe)
                print("result saved")


if __name__ == "__main__":
    # mvc_list = [0.00022255, 0.00022255, 0.00022255,  # Remi
    #             0.00064176,
    #             0.00029489,
    #             0.00063796,
    #             0.00081127, 0.00081127,
    #             0.00016129, 0.00016129, 0.00016129,
    #             0.00065126,
    #             0.00034388,
    #             0.00024886,
    #             0.00013451
    #             ]
    # mvc_list = [0.00021133, 0.00021133 , 0.00021133 ,  # Mathis
    #             0.00055241,
    #             0.00016541 ,
    #             0.00023318,
    #             0.0007307 , 0.0007307 ,
    #             0.00025902, 0.00025902, 0.00025902,
    #             0.00039303,
    #             0.000239,
    #             # 0.00045,
    #             0.00010913
    #             ]
    mvc_list = [0.00015725, 0.00015725, 0.00015725,  # Jules
                0.000574,
                0.00081538,
                0.00056362,
                0.00060177, 0.00060177,
                0.00038563, 0.00038563, 0.00038563,
                0.00031154,
                0.00064221,
                # 0.0008,
                0.00057983
                ]
    subject = "Jules"
    data_dir = f"data/test_09_12_21/{subject}/"
    offline_path = data_dir + 'test_abd'
    result_dir = subject
    is_mhe = True
    configuration_dic = {
        # "model_path": data_dir + f"Wu_Shoulder_Model_mod_wt_wrapp_{subject}_scaled_with_mot.bioMod",
        "model_path": data_dir + f"Wu_Shoulder_Model_mod_wt_wrapp_{subject}.bioMod",
        "mhe_time": 0.15,
        "interpol_factor": 1,
        "use_torque": True,
        "use_excitation": False,
        "save_results": True,
        "track_emg": True,
        "init_w_kalman": False,
        "kin_data_to_track": "markers",
        "mvc_list": mvc_list,
        'muscle_track_idx': [14, 25, 26,  # PEC
                             13,  # DA
                             15,  # DM
                             21,  # DP
                             23, 24,  # bic
                             28, 29, 30,  # tri
                             10,  # TRAPsup
                             2,  # TRAPmed
                             # 3,  # TRAPinf
                             27  # Lat
                             ],
        "result_dir": result_dir,
        "is_mhe": is_mhe,
        "ns_full": 600,
    }

    if configuration_dic["kin_data_to_track"] == 'markers':
        configuration_dic["exp_freq"] = 15
    else:
        configuration_dic["exp_freq"] = 20
    weights = configure_weights(track_emg=configuration_dic["track_emg"],
                                is_mhe=is_mhe,
                                kin_data=configuration_dic["kin_data_to_track"]
                                )

    configuration_dic["weights"] = weights

    variables_dic = {
        "plot_force_freq": 10,
        "emg_rate": 2000,
        "markers_rate": 100,
        "plot_q_freq": 20,
        "print_lvl": 1,
    }
    # data_to_show = ["force", "q"]
    data_to_show = None
    # server_ip = "192.168.1.211"
    server_ip = "127.0.0.1"
    server_port = 50000
    MHE = MuscleForceEstimator(configuration_dic)
    MHE.run(variables_dic,
            server_ip,
            server_port,
            data_to_show,
            test_offline=True,
            offline_file=offline_path,
            data_process=False  # get data in multiprocessing, use if lot of threads on computer

            )
    # print(sol.controls)
    # sol.animate(mesh=True)
