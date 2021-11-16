from biosiglive.client import Client
from biosiglive.data_plot import init_plot_force, init_plot_q
import multiprocessing as mp
import scipy.io as sio
from mhe.ocp import (
    prepare_mhe,
    get_reference_data,
    force_func,
    define_objective,
    update_mhe,
    prepare_short_ocp
)
from bioptim import InitialGuess, InterpolationType
import numpy as np
from mhe.utils import check_and_adjust_dim, update_plot
import biorbd_casadi as biorbd
from time import strftime, time, sleep
import bioviz

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
        self.markers_rate = 100
        self.emg_rate = 2000
        self.get_names = False
        self.get_kalman = True

        self.p_q, self.win_q, self.app_q, self.box_q = [], [], [], []
        self.p_force, self.win_force, self.app_force, self.box_force = [], [], [], []
        self.plot_force_ratio, self.plot_q_ratio = 0, 0
        self.print_lvl = 1
        self.plot_q_freq, self.plot_force_freq = self.exp_freq, 10
        self.force_to_plot, self.q_to_plot = [], []
        self.count_p_f, self.count_p_q = [], []
        self.mvc_list = []

        for key in conf.keys():
            self.__dict__[key] = conf[key]

        self.T_mhe = 1 / self.exp_freq * self.ns_mhe  # Compute the new time of OCP
        self.nbQ, self.nbMT = biorbd_model.nbQ(), biorbd_model.nbMuscles()
        self.nbGT = biorbd_model.nbGeneralizedTorque() if self.use_torque else 0
        self.current_time = strftime("%Y%m%d-%H%M")

        # def _init_mhe(self):

        self.data_to_get = []
        # if self.kin_data_to_track == "markers":
        self.data_to_get.append("markers")
        # elif self.kin_data_to_track == "q":
        #     self.data_to_get.append("q")
        # if self.track_emg:
        self.data_to_get.append("emg")
        if self.test_offline:
            self.x_ref, self.markers_target, self.muscles_target = get_reference_data(offline_path)
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
        # self.muscles_target = np.zeros((self.nbMT, self.ns_mhe))
        # self.markers_target = np.zeros((3, biorbd_model.nbMarkers(), self.ns_mhe + 1))
        # self.x_ref = np.zeros((self.nbQ, self.ns_mhe + 1))
        # X_est = np.ndarray((self.nbQ * 2 + self.nbMT, 1)) if self.use_excitation else np.ndarray((self.nbQ * 2, 1))
        self.kin_target = self.markers_target if self.kin_data_to_track == "markers" else self.x_ref[: self.nbQ, :]
        self.markers_ratio = int(self.markers_rate / self.exp_freq)
        self.EMG_ratio = int(self.emg_rate * self.markers_ratio / self.markers_rate)
        self.rt_ratio = self.markers_ratio
        self.muscle_names = []
        for i in range(biorbd_model.nbMuscles()):
            self.muscle_names.append(biorbd_model.muscleNames()[i].to_string())
        self.dof_names = []
        for i in range(biorbd_model.nbQ()):
            self.dof_names.append(biorbd_model.nameDof()[i].to_string())

        # define some variables
        self.var, self.server_ip, self.server_port, self.data_to_show = {}, [], [], []

        # multiprocess stuffs
        manager = mp.Manager()
        self.plot_queue = manager.Queue()
        self.data_queue = manager.Queue()
        self.process = mp.Process
        self.plot_event = mp.Event()

        if self.track_emg:
            weight = {
                "track_markers": 1000000,
                "track_q": 1000000,
                "min_control": 100,
                "min_dq": 10,
                "min_q": 1,
                "min_torque": 10,
                "min_act": 10,
                "track_emg": 10,
            }
        else:
            weight = {"track_markers": 10000000,
                      "track_q": 1000000, "min_dq": 1, "min_q": 1, "min_torque": 100, "min_act": 1}
            if self.use_excitation:
                weight["min_control"] = 10
            else:
                weight["min_control"] = 100

        objectives = define_objective(
            weights=weight,
            use_excitation=self.use_excitation,
            use_torque=self.use_torque,
            track_emg=self.track_emg,
            muscles_target=self.muscles_target[:, :self.ns_mhe],
            kin_target=self.kin_target[:, :self.ns_mhe + 1],
            biorbd_model=biorbd_model,
            kin_data_to_track=self.kin_data_to_track,
        )

        # Build the graph
        self.mhe, self.solver = prepare_mhe(
            biorbd_model,
            objectives,
            window_len=self.ns_mhe,
            window_duration=self.T_mhe,
            x0=self.x_ref,
            use_torque=self.use_torque,
            use_excitation=self.use_excitation,
            nb_threads=8,
        )

        nlp = self.mhe.nlp[0]
        self.get_force = force_func(biorbd_model, nlp, use_excitation=self.use_excitation)
        self.force_est = np.ndarray((biorbd_model.nbMuscles(), 1))

    # @staticmethod
    def get_data(self, t):
        self.plot_event.wait()
        data_to_get = self.data_to_get
        stream_frequency = 2 * self.exp_freq
        while True:
            tic = time()
            try:
                self.data_queue.get_nowait()
            except:
                pass
                # self.data_count.value = 0
            nb_of_data = self.ns_mhe + 1  # if t == 0 else 2
            vicon_client = Client(self.server_ip, self.server_port, type="TCP")
            data = vicon_client.get_data(
                data_to_get,
                read_frequency=stream_frequency,
                nb_of_data_to_export=nb_of_data,
                nb_frame_of_interest=self.ns_mhe,
                get_names=self.get_names,
                get_kalman=self.get_kalman,
                norm_emg=True,
                mvc_list=self.mvc_list
            )
            self.data_queue.put_nowait(data)
            toc = time() - tic
            if 1/stream_frequency - 1/toc > 0:
                sleep(1/stream_frequency - 1/toc)
            else:
                print("Take too much time to get data")
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
        if test_offline is not True:
            proc_get_data = self.process(name="data", target=MuscleForceEstimator.get_data, args=(self,))
            proc_get_data.start()

        proc_mhe = self.process(name="mhe", target=MuscleForceEstimator.run_mhe, args=(self, var, server_ip, server_port, data_to_show))
        proc_mhe.start()

        if self.data_to_show is not None:
            proc_plot = self.process(name="plot", target=MuscleForceEstimator.run_plot, args=(self,))
            proc_plot.start()

        proc_mhe.join()
        if test_offline is not True:
            proc_get_data.join()

        if self.data_to_show is not None:
            proc_plot.join()

    def run_plot(self):
        for data in self.data_to_show:
            if data == "force":
                print("should plot")
                self.p_force, self.win_force, self.app_force = init_plot_force(self.nbMT, plot_type='progress_bar')
            elif data == "q":
                # self.p_q, self.win_q, self.app_q, self.box_q = init_plot_q(self.nbQ, self.dof_names)
                import bioviz
                self.b = bioviz.Viz(model_path=self.model_path,
                                    show_global_center_of_mass=False,
                                    show_markers=True,
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
                from mhe.utils import update_plot
                update_plot(self, data["t"], data["force_est"], data["q_est"])

    def run_mhe(self, var, server_ip, server_port, data_to_show):
        self.plot_event.wait()
        self.data_to_show = data_to_show
        for key in var.keys():
            if key in self.__dict__:
                self.__setattr__(key, var[key])
            else:
                raise RuntimeError(f'{key} is not a variable of the class')

        if self.test_offline:
            x_ref, markers_ref, muscles_ref = get_reference_data(self.offline_file)
            offline_data = [x_ref, markers_ref, muscles_ref]
        else:
            offline_data = None
        self.model = biorbd.Model(self.model_path)
        initial_time = time()
        muscle_track_idx = [0, 1, 2, 3, 4, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26]

        final_sol = self.mhe.solve(
            lambda mhe, i, sol: update_mhe(mhe, i, sol, self,
                                           initial_time=initial_time,
                                           muscle_track_idx=muscle_track_idx,
                                           offline_data=offline_data),
                                           # export_options={'frame_to_export': self.ns_mhe-1},
                                           solver=self.solver
        )
        # final_sol.graphs()
        final_sol.animate()
        # from bioptim import Shooting
        # from bioptim import Solution
        # ns = 149
        # final_time = ns/self.exp_freq
        # short_ocp = prepare_short_ocp(self.model_path, final_time=final_time, n_shooting=ns)
        # x_init_guess = InitialGuess(final_sol.states["all"], interpolation=InterpolationType.EACH_FRAME)
        # u_init_guess = InitialGuess(final_sol.controls["all"], interpolation=InterpolationType.EACH_FRAME)
        # sol = Solution(short_ocp, [x_init_guess, u_init_guess])
        # sol_int = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, use_scipy_integrator=True)
        # print(sol_int.states['q'][0, :])
        # import matplotlib.pyplot as plt
        # for i in range(sol_int.states['q'].shape[0]):
        #     plt.subplot(5, 5, i + 1)
        #     plt.plot(sol_int.states['q'][i, :])
        #     plt.plot(final_sol.states['q'][i, :])
        # plt.show()
        return final_sol


if __name__ == "__main__":
    # sleep(5)
    mvc_list = sio.loadmat("MVC_18_11_2021/MVC.mat")["MVC_list_max"]
    configuration_dic = {
        "model_path": "models/wu_model.bioMod",
        "ns_mhe": 7,
        "exp_freq": 30,
        "use_torque": True,
        "use_excitation": False,
        "save_results": True,
        "track_emg": True,
        "init_w_kalman": True,
        "kin_data_to_track": "q",
        "mvc_list": mvc_list
    }
    if configuration_dic["kin_data_to_track"] == 'markers':
        configuration_dic["exp_freq"] = 30
    else:
        configuration_dic["exp_freq"] = 30


    variables_dic = {
        "plot_force_freq": 10,
        "emg_rate": 2000,
        "markers_rate": 100,
        "plot_q_freq": 20,
        "print_lvl": 0,
    }
    data_to_show = ["force"]
    # data_to_show = None
    server_ip = "127.0.0.1"
    server_port = 50001
    offline_path = "data/data_09_2021/test_abd.mat"
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
