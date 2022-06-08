"""
This script is the main script for the project. It is used to run the mhe solver and visualize the estimated data.
"""
import os.path
import shutil

from biosiglive.streaming.client import Client
import multiprocessing as mp
from mhe.ocp import *
from mhe.utils import *
from biosiglive.gui.plot import LivePlot


class MuscleForceEstimator:
    """
    This class is used to define the muscle force estimator.
    """

    def __init__(self, *args):
        """
        Initialize the muscle force estimator.

        Parameters
        ----------
        args : dict
            Dictionary of configuration to initialize the estimator.
        """
        conf = check_and_adjust_dim(*args)
        self.model_path = conf["model_path"]
        biorbd_model = biorbd.Model(self.model_path)
        self.use_torque = False
        self.save_results = True
        self.track_emg = False
        self.data_to_show = [""]
        self.kin_data_to_track = None
        self.test_offline = False
        self.offline_file = None
        self.plot_delay = []

        # Variables of the problem
        self.exp_freq = 35
        self.ns_mhe = 0
        self.mhe_time = 0.1
        self.markers_rate = 100
        self.emg_rate = 2000
        self.get_names = False
        self.get_kalman = True
        self.offline_data = None
        self.muscle_track_idx = []
        self.solver_options = {}
        # define some variables
        self.var, self.server_ip, self.server_port, self.data_to_show = {}, None, None, []

        # multiprocess stuffs
        manager = mp.Manager()
        self.data_count = mp.Value("i", 0)
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
        self.ns_full = None
        self.init_n = 0
        self.final_n = None
        self.result_file_name = None
        self.markers_target, self.muscles_target, self.x_ref, self.kin_target = None, None, None, None
        self.n_loop = 0
        self.mhe, self.solver, self.get_force, self.force_est = None, None, None, None
        self.model = None
        self.b = None
        self.frame_to_save = 0

        # Use the configuration dictionary to initialize the muscle force estimator parameters
        for key in conf.keys():
            self.__dict__[key] = conf[key]

        self.T_mhe = self.mhe_time
        self.ns_mhe = int(self.T_mhe * self.markers_rate * self.interpol_factor)
        self.slide_size = int(((self.markers_rate * self.interpol_factor) / self.exp_freq))
        # self.slide_size = 1
        self.nbQ, self.nbMT = biorbd_model.nbQ(), biorbd_model.nbMuscles()
        self.nbGT = biorbd_model.nbGeneralizedTorque() if self.use_torque else 0
        self.current_time = strftime("%Y%m%d-%H%M")
        self.data_to_get = []
        self.data_to_get.append("markers")
        self.data_to_get.append("emg")

        self.markers_ratio = 1
        self.EMG_ratio = 1
        self.rt_ratio = self.markers_ratio
        self.muscle_names = []
        for i in range(biorbd_model.nbMuscles()):
            self.muscle_names.append(biorbd_model.muscleNames()[i].to_string())
        self.dof_names = []
        for i in range(biorbd_model.nbQ()):
            self.dof_names.append(biorbd_model.nameDof()[i].to_string())

    def prepare_problem_init(self):
        """
        Prepare the mhe problem.
        """

        biorbd_model = biorbd.Model(self.model_path)
        self.data_to_get = []
        self.data_to_get.append("markers")
        self.data_to_get.append("emg")
        u0 = None
        if self.test_offline:
            x_ref, markers_target, muscles_target = get_reference_data(self.offline_file)
            self.offline_data = [x_ref, markers_target, muscles_target]

        else:
            nb_of_data = int(self.ns_mhe / self.interpol_factor) + 1
            vicon_client = Client(self.server_ip, self.server_port, type="TCP")
            data = vicon_client.get_data(
                self.data_to_get,
                read_frequency=self.markers_rate,
                nb_of_data_to_export=nb_of_data,
                get_names=self.get_names,
                get_kalman=self.get_kalman,
            )

            x_ref = np.array(data["kalman"])
            markers_target = np.array(data["markers"])
            muscles_target = np.array(data["emg"])

        window_len = self.ns_mhe
        window_duration = self.T_mhe
        self.x_ref, self.markers_target, muscles_target = interpolate_data(
            self.interpol_factor, x_ref, muscles_target, markers_target
        )

        self.muscles_target = muscle_mapping(
            muscles_target_tmp=muscles_target, mvc_list=mvc_list, muscle_track_idx=self.muscle_track_idx
        )[:, :window_len]
        self.kin_target = (
            self.markers_target[:, :, : window_len + 1]
            if self.kin_data_to_track == "markers"
            else self.x_ref[:self.nbQ, : window_len + 1]
        )

        for i in range(biorbd_model.nbMuscles()):
            self.muscle_names.append(biorbd_model.muscleNames()[i].to_string())
        previous_sol = np.concatenate((self.x_ref[:, : window_len + 1], np.zeros((self.x_ref.shape[0], window_len + 1))))

        objectives = define_objective(
            weights=self.weights,
            use_torque=self.use_torque,
            track_emg=self.track_emg,
            muscles_target=self.muscles_target,
            kin_target=self.kin_target,
            biorbd_model=biorbd_model,
            previous_sol=previous_sol,
            kin_data_to_track=self.kin_data_to_track,
            muscle_track_idx=self.muscle_track_idx,
        )

        self.mhe, self.solver = prepare_problem(
            self.model_path,
            objectives,
            window_len=window_len,
            window_duration=window_duration,
            x0=self.x_ref,
            u0=u0,
            use_torque=self.use_torque,
            nb_threads=8,
            solver_options=self.solver_options,
            use_acados=True,
        )
        self.get_force = force_func(biorbd_model)
        self.force_est = np.ndarray((biorbd_model.nbMuscles(), 1))

    def get_data(self):
        """
        Get data from the vicon server.

        Returns
        -------
        Asked data
        """
        data_to_get = self.data_to_get
        nb_of_data = self.ns_mhe + 1
        vicon_client = Client(self.server_ip, self.server_port, type="TCP")
        data = vicon_client.get_data(
            data_to_get,
            read_frequency=self.exp_freq,
            nb_of_data_to_export=nb_of_data,
            get_names=self.get_names,
            get_kalman=self.get_kalman,
        )
        return data

    def run(
        self,
        var: dict,
        server_ip: str,
        server_port: int,
        data_to_show: list = None,
        test_offline: bool = False,
        offline_file: str = None,
    ):
        """
        Run the whole multiprocess program.

        Parameters
        ----------
        var : dict
            Dictionary containing the parameters of the problem.
        server_ip : str
            IP of the vicon server.
        server_port : int
            Port of the vicon server.
        data_to_show : list, optional
            List of data to show. The default is None.
        test_offline : bool, optional
            If True, the program will run in offline mode. The default is False.
        offline_file : str, optional
            Path to the offline file. The default is None.
        """
        self.var = var
        self.server_ip = server_ip
        self.server_port = server_port
        self.data_to_show = data_to_show
        self.test_offline = test_offline
        self.offline_file = offline_file
        proc_plot = None
        if self.test_offline and not self.offline_file:
            raise RuntimeError("Please provide a data file to run offline program")

        if self.data_to_show:
            proc_plot = self.process(name="plot", target=MuscleForceEstimator.run_plot, args=(self,))
            proc_plot.start()

        proc_mhe = self.process(name="mhe", target=MuscleForceEstimator.run_mhe, args=(self, var, data_to_show))
        proc_mhe.start()
        if self.data_to_show:
            proc_plot.join()
        proc_mhe.join()

    def run_plot(self):
        """
        Run the plot function.
        """
        data = None
        for data_to_show in self.data_to_show:
            if data_to_show == "force":
                self.p_force, self.win_force, self.app_force = LivePlot.init_plot_windows(self.nbMT)

            if data_to_show == "q":
                import bioviz

                self.b = bioviz.Viz(
                    model_path=self.model_path,
                    show_global_center_of_mass=False,
                    show_markers=True,
                    show_floor=False,
                    show_gravity_vector=False,
                    show_muscles=False,
                    show_segments_center_of_mass=False,
                    show_local_ref_frame=False,
                    show_global_ref_frame=False,
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
            except mp.Queue.queue.Empty:
                is_working = False

            if is_working:
                plot_delay = update_plot(self, data["force_est"], data["q_est"], init_time=data["init_time_frame"])
                dic = {"plot_delay": plot_delay}
                save_results(dic, self.current_time, result_dir=self.result_dir, file_name_prefix="plot_delay_")

    def run_mhe(self, var: dict, data_to_show: list):
        """
        Run the mhe solver.

        Parameters
        ----------
        var : dict
            Dictionary containing the parameters of the problem.
        data_to_show : list
            List of data to show.
        """
        if os.path.isdir("c_generated_code"):
            shutil.rmtree("c_generated_code")
        self.prepare_problem_init()
        if data_to_show:
            self.plot_event.wait()
        for key in var.keys():
            if key in self.__dict__:
                self.__setattr__(key, var[key])
            else:
                raise RuntimeError(f"{key} is not a variable of the class")

        self.model = biorbd.Model(self.model_path)
        initial_time = time()

        sol = self.mhe.solve(
            lambda mhe, i, sol: update_mhe(
                mhe, i, sol, self, initial_time=initial_time, offline_data=self.offline_data
            ),
            export_options={"frame_to_export": self.frame_to_save},
            solver=self.solver,
        )

        import matplotlib.pyplot as plt
        q_est = sol.states["q"][:, :]
        qdot_est = sol.states["qdot"][:, :]
        qdot_dif = np.copy(qdot_est)

        kalman = self.mhe.kalman
        def finite_difference(data):
            t = np.linspace(0, data.shape[0] / 100, data.shape[0])
            y = data
            dydt = np.gradient(y, t)
            return dydt
        for i in range(qdot_est.shape[0]):
            qdot_dif[i, :] = finite_difference(q_est[i, :])

        plt.figure("q_dot")
        # t_est = np.linspace(0, q_est.shape[1]/35, q_est.shape[1])
        # t = np.linspace(0, kalman.shape[1]/100, kalman.shape[1])

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(qdot_est[i, :] * 180 / np.pi, label="qdot_est")
            plt.plot(qdot_dif[i, :] * 180 / np.pi, label="qdot_dif")
            # plt.plot(t, qdot[i, :] * 180 / np.pi, label="qdotkalman")
            # plt.title(model.nameDof()[i].to_string())

        plt.figure("q")
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(q_est[i, :] * 180 / np.pi, label="q_est")
            # plt.plot(kalman[i, :] * 180 / np.pi, label="kalman")
            # plt.plot(self.x_ref[i, :] * 180 / np.pi, label="x_ref")
            # plt.plot(t, qdot[i, :] * 180 / np.pi, label="qdotkalman")
            # plt.title(model.nameDof()[i].to_string())
        plt.legend()
        sol.graphs()
        plt.show()


if __name__ == "__main__":
    subject = f"subject_1"
    data_dir = f"data_final/{subject}/"

    mvc = sio.loadmat(data_dir + f"MVC_{subject}.mat")["MVC_list_max"][0]
    mvc_list = [
        mvc[0],
        mvc[0],
        mvc[0],  # MVC Pectoralis sternalis
        mvc[1],  # MVC Deltoid anterior
        mvc[2],  # MVC Deltoid medial
        mvc[3],  # MVC Deltoid posterior
        mvc[4],
        mvc[4],  # MVC Biceps brachii
        mvc[5],
        mvc[5],
        mvc[5],  # MVC Triceps brachii
        mvc[6],  # MVC Trapezius superior
        mvc[7],  # MVC Trapezius medial
        mvc[8],  # MVC Trapezius inferior
        mvc[9],  # MVC Latissimus dorsi
    ]

    result_dir = data_dir
    trials = ["abd"]
    configs = ["mhe"]  # , "mhe"]

    for config in configs:
        for trial in trials:
            offline_path = result_dir + f"{trial}"
            file_name = f"{trial}_result_test"
            solver_options = {"sim_method_jac_reuse": 1, "nlp_solver_step_length": 0.5, "levenberg_marquardt": 100.0}
            # solver_options = {}

            configuration_dic = {
                "model_path": data_dir + f"model_{subject}_scaled.bioMod",
                "mhe_time": 0.1,
                "interpol_factor": 2,
                "use_torque": False,
                "save_results": True,
                "track_emg": True,
                "kin_data_to_track": "q",
                "mvc_list": mvc_list,
                "exp_freq": 35,
                "muscle_track_idx": [
                    14,
                    25,
                    26,  # MVC Pectoralis sternalis
                    13,  # MVC Deltoid anterior
                    15,  # MVC Deltoid medial
                    21,  # MVC Deltoid posterior
                    23,
                    24,  # MVC Biceps brachii
                    28,
                    29,
                    30,  # MVC Triceps brachii
                    10,  # MVC Trapezius superior
                    2,  # MVC Trapezius medial
                    3,  # MVC Trapezius inferior
                    27,  # MVC Latissimus dorsi
                ],
                "result_dir": result_dir,
                "result_file_name": file_name,
                "solver_options": solver_options,
                "weights": configure_weights(),
                "frame_to_save": 0,
            }

            variables_dic = {"print_lvl": 1}  # print level 0 = no print, 1 = print information

            # data_to_show = ["force", "q"]
            data_to_show = None
            server_ip = "127.0.0.1"
            server_port = 50000
            MHE = MuscleForceEstimator(configuration_dic)
            MHE.run(variables_dic, server_ip, server_port, data_to_show, test_offline=True, offline_file=offline_path)
