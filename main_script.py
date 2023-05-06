"""
This script is the main script for the project. It is used to run the mhe solver and visualize the estimated data.
"""
import os.path
import shutil
from biosiglive.streaming.client import Message
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
        biorbd_model = BiorbdModel(self.model_path)
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
        self.message = None
        self.vicon_client = None
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
        self.mvc_list = None
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
        self.save_all_frame = True

        # Use the configuration dictionary to initialize the muscle force estimator parameters
        for key in conf.keys():
            self.__dict__[key] = conf[key]

        self.T_mhe = self.mhe_time
        self.ns_mhe = int(self.T_mhe * self.markers_rate * self.interpol_factor)
        self.slide_size = int(((self.markers_rate * self.interpol_factor) / self.exp_freq))
        self.nbQ, self.nbMT = biorbd_model.nb_q, biorbd_model.nb_muscles
        self.nbGT = biorbd_model.nb_tau if self.use_torque else 0
        self.current_time = strftime("%Y%m%d-%H%M")
        self.data_to_get = []
        self.data_to_get.append("markers")
        self.data_to_get.append("emg")

        self.markers_ratio = 1
        self.EMG_ratio = 1
        self.rt_ratio = self.markers_ratio
        self.muscle_names = []
        for i in range(biorbd_model.nb_muscles):
            self.muscle_names.append(biorbd_model.muscle_names[i])
        self.dof_names = []
        for i in range(biorbd_model.nb_q):
            self.dof_names.append(biorbd_model.name_dof[i])

    def prepare_problem_init(self):
        """
        Prepare the mhe problem.
        """

        biorbd_model = BiorbdModel(self.model_path)
        self.data_to_get = []
        self.data_to_get.append("markers")
        self.data_to_get.append("emg")
        if self.test_offline:
            x_ref, markers_target, muscles_target = get_data(offline=True, offline_file_path=self.offline_file)
            self.offline_data = [x_ref, markers_target, muscles_target]

        else:
            nb_of_data = int(self.ns_mhe / self.interpol_factor) + 1
            self.message = Message(
                command=self.data_to_get,
                read_frequency=self.exp_freq,
                nb_frame_to_get=nb_of_data,
                get_raw_data=False,
                kalman=True,
                ratio=1,
            )
            data = get_data(ip=self.server_ip, port=self.server_port, message=self.message)
            x_ref = np.array(data["kalman"])
            markers_target = np.array(data["markers"])[:, :, :]
            muscles_target = np.array(data["emg_proc"])

        window_len = self.ns_mhe
        window_duration = self.T_mhe
        self.x_ref, self.markers_target, muscles_target = interpolate_data(
            self.interpol_factor, x_ref, muscles_target, markers_target
        )
        self.muscles_target = muscle_mapping(
            muscles_target_tmp=muscles_target, muscle_track_idx=self.muscle_track_idx, mvc_list=self.mvc_list
        )[:, :window_len]
        self.kin_target = (
            self.markers_target[:, :, : window_len + 1]
            if self.kin_data_to_track == "markers"
            else self.x_ref[: self.nbQ, : window_len + 1]
        )

        for i in range(biorbd_model.nb_muscles):
            self.muscle_names.append(biorbd_model.muscle_names[i])
        if self.x_ref.shape[0] != biorbd_model.nb_q * 2:
            previous_sol = np.concatenate(
                (self.x_ref[:, : window_len + 1], np.zeros((self.x_ref.shape[0], window_len + 1)))
            )
        else:
            previous_sol = self.x_ref[:, : window_len + 1]
        muscle_init = np.ones((biorbd_model.nb_muscles, self.ns_mhe)) * 0.1
        count = 0
        for i in self.muscle_track_idx:
            muscle_init[i, :] = self.muscles_target[count, : self.ns_mhe]
            count += 1
        u0 = np.concatenate((muscle_init, np.zeros((biorbd_model.nb_q, self.ns_mhe))))
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
        self.force_est = np.ndarray((biorbd_model.nb_muscles, 1))

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
        self.all_plot = LivePlot()
        for data_to_show in self.data_to_show:
            if data_to_show == "force":
                self.all_plot.add_new_plot(
                    plot_name="Muscle force",
                    plot_type="progress_bar",
                    nb_subplot=self.nbMT,
                    channel_names=self.muscle_names,
                    unit="N",
                )
                self.rplt_force, self.layout_force, self.app_force = self.all_plot.init_plot_window(
                    self.all_plot.plot[0]
                )
            if data_to_show == "q":
                self.all_plot.msk_model = self.model_path
                self.all_plot.add_new_plot(plot_type="skeleton")
                self.all_plot.set_skeleton_plot_options(show_floor=False)
                n_plot = 0 if not "force" in self.data_to_show else 1
                self.all_plot.init_plot_window(self.all_plot.plot[n_plot])

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

        self.model = BiorbdModel(self.model_path)
        initial_time = time()
        sol = self.mhe.solve(
            lambda mhe, i, sol: update_mhe(
                mhe, i, sol, self, initial_time=initial_time, offline_data=self.offline_data
            ),
            export_options={"frame_to_export": self.frame_to_save},
            solver=self.solver,
        )


if __name__ == "__main__":
    data_dir = f"/home/amedeo/Documents/programmation/code_paper_mhe/data/data_final_new/subject_3/C3D/"
    result_dir = "results/results_w9/"
    trials = [
        "data_abd_sans_poid",
        "data_abd_poid_2kg",
        # "data_cycl_poid_2kg",
        # "data_flex_poid_2kg",
        # "data_flex_sans_poid",
        # "data_cycl_sans_poid",
    ]
    # configs = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
    # exp_freq = [43, 38, 37, 34, 29, 27, 25, 24, 22]
    configs = [0.07]
    exp_freq = [30]
    for c, config in enumerate(configs):
        for trial in trials:
            offline_path = data_dir + f"{trial}"
            file_name = f"{trial}_result_duration_{config}_test"

            solver_options = {
                "sim_method_jac_reuse": 1,
                "levenberg_marquardt": 50.0,
                "nlp_solver_step_length": 0.9,
                "qp_solver_iter_max": 500,
            }
            if "2k" in trial:
                model = f"data/wu_scaled_2kg.bioMod"
            else:
                model = f"data/wu_scaled.bioMod"

            configuration_dic = {
                "model_path": model,
                "mhe_time": config,
                "interpol_factor": 2,
                "use_torque": True,
                "save_results": True,
                "track_emg": True,
                "kin_data_to_track": "markers",
                "exp_freq": exp_freq[c],
                "muscle_track_idx": [
                    14,
                    23,
                    24,  # MVC Pectoralis sternalis
                    13,  # MVC Deltoid anterior
                    15,  # MVC Deltoid medial
                    16,  # MVC Deltoid posterior
                    26,
                    27,  # MVC Biceps brachii
                    28,
                    29,
                    30,  # MVC Triceps brachii
                    11,
                    1,  # MVC Trapezius superior bis
                    2,  # MVC Trapezius medial
                    3,  # MVC Trapezius inferior
                    25,  # MVC Latissimus dorsi
                ],
                "result_dir": result_dir,
                "result_file_name": file_name,
                "solver_options": solver_options,
                "weights": configure_weights(),
                "frame_to_save": 0,
                "save_all_frame": True,
            }
            variables_dic = {"print_lvl": 1}  # print level 0 = no print, 1 = print information
            data_to_show = None  # ["q", "force"]
            server_ip = "192.168.1.211"
            server_port = 50000
            MHE = MuscleForceEstimator(configuration_dic)
            MHE.run(variables_dic, server_ip, server_port, data_to_show, test_offline=True, offline_file=offline_path)
