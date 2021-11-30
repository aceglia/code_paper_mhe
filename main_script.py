from biosiglive.client import Client
# from biosiglive.data_plot import init_plot_force, init_plot_q
import multiprocessing as mp
# import scipy.io as sio
# from mhe.ocp import (
#     prepare_mhe,
#     get_reference_data,
#     force_func,
#     define_objective,
#     update_mhe,
#     prepare_short_ocp
# )
# from bioptim import InitialGuess, InterpolationType
# import numpy as np
# from mhe.utils import check_and_adjust_dim, update_plot
import biorbd_casadi as biorbd
# from time import strftime, time, sleep
# import bioviz
from biosiglive.data_plot import init_plot_force, init_plot_q, update_plot_force, update_plot_q
from mhe.ocp import *
from mhe.utils import *

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

        # Variables of the problem
        self.exp_freq = 30
        self.ns_mhe = 7
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

        for key in conf.keys():
            self.__dict__[key] = conf[key]

        self.T_mhe = self.mhe_time

        # self.T_mhe = round(1 / self.exp_freq, 2)
        # self.T_mhe = np.round((1 / self.exp_freq, 2))# * self.ns_mhe), 2)  # Compute the new time of OCP
        print(self.T_mhe)
        print(self.exp_freq)

        self.ns_mhe = int(self.T_mhe * self.markers_rate* self.interpol_factor)
        print(self.ns_mhe)
        self.slide_size = int((1 / self.exp_freq) / (1 /self.markers_rate) * self.interpol_factor)
        print(self.slide_size)
        self.nbQ, self.nbMT = biorbd_model.nbQ(), biorbd_model.nbMuscles()
        self.nbGT = biorbd_model.nbGeneralizedTorque() if self.use_torque else 0
        self.current_time = strftime("%Y%m%d-%H%M")
        self.data_to_get = []
        self.data_to_get.append("markers")
        self.data_to_get.append("emg")
        self.muscle_track_idx = [13, 15, 21, 10, 1, 2, 27, 14, 25, 26, 23, 24, 28, 29, 30]

        self.markers_ratio = 1 #int(self.markers_rate / self.exp_freq)
        print(self.markers_ratio)
        self.EMG_ratio = 20 #int(self.emg_rate / self.exp_freq)
        print(self.EMG_ratio)
        self.rt_ratio = self.markers_ratio
        self.muscle_names = []
        for i in range(biorbd_model.nbMuscles()):
            self.muscle_names.append(biorbd_model.muscleNames()[i].to_string())
        self.dof_names = []
        for i in range(biorbd_model.nbQ()):
            self.dof_names.append(biorbd_model.nameDof()[i].to_string())

    def prepare_problem(self):
        biorbd_model = biorbd.Model(self.model_path)
        # def _init_mhe(self):

        self.data_to_get = []
        # if self.kin_data_to_track == "markers":
        self.data_to_get.append("markers")
        # elif self.kin_data_to_track == "q":
        #     self.data_to_get.append("q")
        # if self.track_emg:
        self.data_to_get.append("emg")
        from scipy import interpolate

        if offline_path:
            self.x_ref, self.markers_target_tmp, self.muscles_target = get_reference_data(offline_path)
            self.markers_target = np.zeros((3, biorbd_model.nbMarkers(), int(self.markers_target_tmp.shape[2] * self.interpol_factor)))
            x = np.linspace(0, self.x_ref.shape[1] / 100, self.x_ref.shape[1])
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(x, self.x_ref[0, :].T)
            f_x = interpolate.interp1d(x, self.x_ref)
            x_new = np.linspace(0, self.x_ref.shape[1]/100, int(self.x_ref.shape[1] * self.interpol_factor))
            self.x_ref = f_x(x_new)
            # plt.plot(x_new, self.x_ref[0, :].T)
            # plt.show()
            for i in range(3):
                x = np.linspace(0, self.markers_target_tmp.shape[2] / 100, self.markers_target_tmp.shape[2])
                f_mark = interpolate.interp1d(x, self.markers_target_tmp[i, :, :])
                x_new = np.linspace(0, self.markers_target_tmp.shape[2] / 100, int(self.markers_target_tmp.shape[2] * self.interpol_factor))
                self.markers_target[i, :, :] = f_mark(x_new)

            x = np.linspace(0, self.muscles_target.shape[1] / 100, self.muscles_target.shape[1])
            f_mus = interpolate.interp1d(x, self.muscles_target)
            x_new = np.linspace(0, self.muscles_target.shape[1]/100, int(self.muscles_target.shape[1] * self.interpol_factor))
            self.muscles_target = f_mus(x_new)
            # self.markers_target = self.markers_target[:3, :, :]

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
                norm_emg=True,
                mvc_list=self.mvc_list
            )
            self.x_ref = np.array(data["kalman"])
            self.markers_ref = np.array(data["markers"])
            self.muscles_ref = np.array(data["emg"])


        self.kin_target = self.markers_target if self.kin_data_to_track == "markers" else self.x_ref[: self.nbQ, :]
        # self.markers_ratio = int(self.markers_rate / self.exp_freq)
        # self.EMG_ratio = int(self.emg_rate * self.markers_ratio / self.markers_rate)
        # self.rt_ratio = self.markers_ratio
        # self.muscle_names = []
        for i in range(biorbd_model.nbMuscles()):
            self.muscle_names.append(biorbd_model.muscleNames()[i].to_string())
        self.dof_names = []
        for i in range(biorbd_model.nbQ()):
            self.dof_names.append(biorbd_model.nameDof()[i].to_string())

        if self.track_emg:
            weight = {
                "track_markers": 10000000,
                "track_q": 100000,
                "min_control": 1,
                "min_dq": 10,
                "min_q": 1,
                "min_torque": 1000,
                "min_act": 1,
                "track_emg": 100,
            }
        else:
            weight = {"track_markers": 10000000,
                      "track_q": 1000000, "min_dq": 10, "min_q": 1, "min_torque": 1000, "min_act": 1}
            if self.use_excitation:
                weight["min_control"] = 10
            else:
                weight["min_control"] = 10

        kin_target = self.kin_target[:, :self.ns_mhe + 1] if self.kin_data_to_track == "q" else self.kin_target[:, :, :self.ns_mhe + 1]

        objectives = define_objective(
            weights=weight,
            use_excitation=self.use_excitation,
            use_torque=self.use_torque,
            track_emg=self.track_emg,
            muscles_target=self.muscles_target[:, :self.ns_mhe],
            kin_target=kin_target,
            biorbd_model=biorbd_model,
            kin_data_to_track=self.kin_data_to_track,
            muscle_track_idx=self.muscle_track_idx
        )

        self.mhe, self.solver = prepare_mhe(
            biorbd_model,
            objectives,
            window_len=self.ns_mhe,
            window_duration=self.T_mhe,
            x0=self.x_ref,
            use_torque=self.use_torque,
            use_excitation=self.use_excitation,
            nb_threads=6,
        )

        nlp = self.mhe.nlp[0]
        self.get_force = force_func(biorbd_model, nlp, use_excitation=self.use_excitation)
        self.force_est = np.ndarray((biorbd_model.nbMuscles(), 1))
    # @staticmethod

    def get_data(self, t):
        # self.plot_event.wait()
        data_to_get = self.data_to_get
        # stream_frequency = 2 * self.exp_freq
        while True:
            # tic = time()
            try:
                self.data_queue.get_nowait()
            except:
                pass
                # self.data_count.value = 0
            nb_of_data = self.ns_mhe + 1  # if t == 0 else 2
            vicon_client = Client(self.server_ip, self.server_port, type="TCP")
            data = vicon_client.get_data(
                data_to_get,
                read_frequency=self.exp_freq,
                nb_of_data_to_export=nb_of_data,
                nb_frame_of_interest=self.ns_mhe,
                get_names=self.get_names,
                get_kalman=self.get_kalman,
                norm_emg=True,
                mvc_list=self.mvc_list
            )
            self.data_queue.put_nowait(data)
            # toc = time() - tic
            # if 1/stream_frequency - 1/toc > 0:
            #     sleep(1/stream_frequency - 1/toc)
            # else:
            #     print("Take too much time to get data")
        # self.data_count.value += 1
        #     if self.data_event.is_set() is not True:
        #         print(self.data_event.is_set())
        #         self.data_event.set()

        # return data

    def run(self, var, server_ip, server_port, data_to_show=None, test_offline=False, offline_file=None):
        self.var = var
        self.server_ip = server_ip
        self.server_port = server_port
        self.data_to_show = data_to_show
        self.test_offline = test_offline
        self.offline_file = offline_file
        proc_plot, proc_get_data = [], []
        t = 0
        if test_offline is not True:
            proc_get_data = self.process(name="data", target=MuscleForceEstimator.get_data, args=(self, t))
            proc_get_data.start()

        if self.data_to_show:
            proc_plot = self.process(name="plot", target=MuscleForceEstimator.run_plot, args=(self,))
            proc_plot.start()

        proc_mhe = self.process(name="mhe", target=MuscleForceEstimator.run_mhe, args=(self, var, server_ip, server_port, data_to_show))
        proc_mhe.start()
        # MuscleForceEstimator.run_mhe(self, var, server_ip, server_port, data_to_show)

        if self.data_to_show:
            proc_plot.join()

        proc_mhe.join()
        if test_offline is not True:
            proc_get_data.join()

    def run_plot(self):
        # print(1)
        # print(self.plot_queue)
        # self.data_to_show = data_to_show
        # initialize plot
        # if data_to_show:
        for data_to_show in self.data_to_show:
            if data_to_show == "force":
                # p_force, win_force, app_force, box_force = init_plot_force(self.nbMT)
                self.p_force, self.win_force, self.app_force = init_plot_force(self.nbMT)

            elif data_to_show == "q":
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
        # sys.stdin = os.fdopen(0)
        # ready = input("Ready to run estimator ? (tape any key to continue)")
        # if ready:
        #     pass
        #     self.data_to_show = data_to_show
        for key in var.keys():
            if key in self.__dict__:
                self.__setattr__(key, var[key])
            else:
                raise RuntimeError(f'{key} is not a variable of the class')

        if self.test_offline:
            x_ref, markers_ref, muscles_ref = get_reference_data(self.offline_file)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(muscles_ref.T)
            # plt.show()
            offline_data = [x_ref, markers_ref, muscles_ref]
        else:
            offline_data = None
        self.model = biorbd.Model(self.model_path)
        initial_time = time()
        # muscle_track_idx = [0, 1, 2, 3, 4, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26]

        final_sol = self.mhe.solve(
            lambda mhe, i, sol: update_mhe(mhe, i, sol, self,
                                           initial_time=initial_time,
                                           muscle_track_idx=self.muscle_track_idx,
                                           offline_data=offline_data),
                                           export_options={'frame_to_export': self.ns_mhe-1},
                                           solver=self.solver
        )
        # final_sol.animate()
        final_sol.graphs()
        return final_sol


if __name__ == "__main__":
    # sleep(5)
    # mvc_list = sio.loadmat(
    #     "/home/amedeo/Documents/programmation/code_paper_mhe/data/test_18_11_21/gregoire/test_1/MVC.mat")["MVC_list_max"].tolist()[0]
    mvc_list = sio.loadmat(
    "/home/amedeo/Documents/programmation/code_paper_mhe/data/test_18_11_21/gregoire/test_1/MVC.mat")["mvc_treat"].tolist()[0]
    configuration_dic = {
        # "model_path": "models/wu_model.bioMod",
        # "model_path": "models/Wu_Shoulder_Model_mod_wt_wrapp_scaled_gregoire.bioMod",
        "model_path": "models/Wu_Shoulder_Model_mod_wt_wrapp_remi.bioMod",
        # "ns_mhe": 7*3,
        "mhe_time": 0.1,
        "interpol_factor": 0.8,
        # "exp_freq": 30,
        "use_torque": True,
        "use_excitation": False,
        "save_results": True,
        "track_emg": True,
        "init_w_kalman": False,
        "kin_data_to_track": "q",
        "mvc_list": mvc_list,
        'muscle_track_idx': [13, 15, 21, 10, 1, 2, 27, 14, 25, 26, 23, 24, 28, 29, 30],
    }
    if configuration_dic["kin_data_to_track"] == 'markers':
        configuration_dic["exp_freq"] = 15
    else:
        configuration_dic["exp_freq"] = 20

    variables_dic = {
        "plot_force_freq": 10,
        "emg_rate": 2000,
        "markers_rate": 100,
        "plot_q_freq": 20,
        "print_lvl": 1,
    }
    # data_to_show = ["force", "q"]
    data_to_show = None
    server_ip = "192.168.1.211"
    server_port = 50001
    # offline_path = "data/data_09_2021/test_abd.mat"
    offline_path = "data/test_18_11_21/gregoire/test_1/test_flex.mat"
    MHE = MuscleForceEstimator(configuration_dic)
    MHE.run(variables_dic,
            server_ip,
            server_port,
            data_to_show,
            test_offline=True,
            offline_file=offline_path

            )
    # print(sol.controls)
    # sol.animate(mesh=True)
