from biosiglive.client import Client
import multiprocessing as mp
from mhe.ocp import *
from mhe.utils import *
from biosiglive.data_plot import init_plot_force


class MuscleForceEstimator:
    def __init__(self, *args):
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
        self.var, self.server_ip, self.server_port, self.data_to_show = {}, [], [], []

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

        for key in conf.keys():
            self.__dict__[key] = conf[key]
        self.T_mhe = self.mhe_time
        self.ns_mhe = int(self.T_mhe * self.markers_rate * self.interpol_factor)
        self.slide_size = int(((self.markers_rate * self.interpol_factor) / self.exp_freq))
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
        biorbd_model = biorbd.Model(self.model_path)
        self.data_to_get = []
        self.data_to_get.append("markers")
        self.data_to_get.append("emg")
        u0 = None
        if self.test_offline:
            x_ref, markers_target, muscles_target = get_reference_data(self.offline_file)
            self.final_n = -1 if not self.final_n else self.final_n
            x_ref, markers_target, muscles_target = (x_ref[:, self.init_n:self.final_n],
                                                     markers_target[:, :, self.init_n:self.final_n],
                                                     muscles_target[:, self.init_n:self.final_n])

            self.offline_data = [x_ref, markers_target, muscles_target]

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
            )

            x_ref = np.array(data["kalman"])
            markers_target = np.array(data["markers"])
            muscles_target = np.array(data["emg"])

        window_len = self.ns_mhe
        window_duration = self.T_mhe

        self.x_ref, self.markers_target, muscles_target = interpolate_data(self.interpol_factor,
                                                                           x_ref,
                                                                           muscles_target,
                                                                           markers_target
                                                                           )

        self.muscles_target = muscle_mapping(muscles_target_tmp=muscles_target,
                                             mvc_list=mvc_list,
                                             muscle_track_idx=self.muscle_track_idx
                                             )[:, :window_len]

        if self.use_torque:
            nbGT = biorbd_model.nbQ()
        else:
            nbGT = 0

        if u0 is None:
            u0 = np.ones((biorbd_model.nbMuscles() + nbGT, window_len)) * 0.1
            c = 0
            for i in range(biorbd_model.nbQ(), biorbd_model.nbMuscles() + nbGT):
                if c in self.muscle_track_idx:
                    idx = self.muscle_track_idx.index(c)
                    u0[c, :] = self.muscles_target[idx, :]
                    c += 1

        self.kin_target = (
            self.markers_target[:, :, : window_len + 1]
            if self.kin_data_to_track == "markers"
            else self.x_ref[: self.nbQ, : window_len + 1]
        )
        # if is_mhe:
        #     self.x_ref = self.x_ref[:, : window_len + 1]

        for i in range(biorbd_model.nbMuscles()):
            self.muscle_names.append(biorbd_model.muscleNames()[i].to_string())

        # Warm start for the full ocp
        objectives = define_objective(
            weights=self.weights,
            use_torque=self.use_torque,
            track_emg=self.track_emg,
            muscles_target=self.muscles_target,
            kin_target=self.kin_target,
            biorbd_model=biorbd_model,
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
        data_to_get = self.data_to_get
        nb_of_data = self.ns_mhe + 1  # if t == 0 else 2
        vicon_client = Client(self.server_ip, self.server_port, type="TCP")
        data = vicon_client.get_data(
            data_to_get,
            read_frequency=self.exp_freq,
            nb_of_data_to_export=nb_of_data,
            nb_frame_of_interest=self.ns_mhe,
            get_names=self.get_names,
            get_kalman=self.get_kalman,
        )
        return data

    def run(
        self, var, server_ip, server_port, data_to_show=None, test_offline=False, offline_file=None
    ):
        self.var = var
        self.server_ip = server_ip
        self.server_port = server_port
        self.data_to_show = data_to_show
        self.test_offline = test_offline
        self.offline_file = offline_file
        if self.test_offline and not self.offline_file:
            raise RuntimeError("Please provide a data file to run offline program")
        proc_plot = []

        if self.data_to_show:
            proc_plot = self.process(name="plot", target=MuscleForceEstimator.run_plot, args=(self,))
            proc_plot.start()

        proc_mhe = self.process(
            name="mhe", target=MuscleForceEstimator.run_mhe, args=(self, var, data_to_show)
        )
        proc_mhe.start()
        if self.data_to_show:
            proc_plot.join()
        proc_mhe.join()

    def run_plot(self):
        for data_to_show in self.data_to_show:
            if data_to_show == "force":
                self.p_force, self.win_force, self.app_force = init_plot_force(self.nbMT)

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
            except:
                is_working = False

            if is_working:
                plot_delay = update_plot(
                    self, data["t"], data["force_est"], data["q_est"], init_time=data["init_time_frame"]
                )
                dic = {"plot_delay": plot_delay}
                save_results(dic, self.current_time, result_dir=self.result_dir, file_name_prefix="plot_delay_")

    def run_mhe(self, var, data_to_show):
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

        final_sol = self.mhe.solve(
            lambda mhe, i, sol: update_mhe(
                mhe,
                i,
                sol,
                self,
                initial_time=initial_time,
                offline_data=self.offline_data,
            ),
            export_options={"frame_to_export": self.ns_mhe - 1},
            solver=self.solver,
        )
        # final_sol.graphs()


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
    # mvc_list = [0.00011193,0.00011193,0.00011193,
    #  0.00195273,
    #  0.0004393,
    #  0.00044208,
    #  0.00081404,0.00081404,
    #  0.00019145,0.00019145,0.00019145,
    #  0.00448458,
    #  0.00171529,
    #  0.00134165,
    #  0.00020328]

    scaled = True
    scal = "_scaled" if scaled else ""
    subject = f"Clara"
    data_dir = f"/home/amedeo/Documents/programmation/data_article/{subject}/"

    mvc = sio.loadmat(data_dir + f"MVC_{subject}.mat")["MVC_list_max"][0]
    mvc_list = [
        mvc[0],
        mvc[0],
        mvc[0],
        mvc[1],
        mvc[2],
        mvc[3],
        mvc[4],
        mvc[4],
        mvc[5],
        mvc[5],
        mvc[5],
        mvc[6],
        mvc[7],
        mvc[8],
        mvc[9],
    ]

    result_dir = data_dir
    # trials = ["abd", "abd_cocon", "flex", "flex_cocon", "cycl","cycl_cocon"]  # , "abd_1_rep", "flex_1_rep", "flex_cocon_1_rep"]
    trials = ["abd"]
    configs = ["mhe"]  # , "mhe"]

    for config in configs:
        for trial in trials:
            offline_path = result_dir + f"{trial}"
            file_name = f"{trial}_result"
            solver_options = {
                "sim_method_jac_reuse": 1,
                "nlp_solver_step_length": 0.5,
                "levenberg_marquardt": 100.0,
            }
            configuration_dic = {"model_path": data_dir + f"Wu_Shoulder_Model_mod_wt_wrapp_{subject}{scal}.bioMod",
                                 "mhe_time": 0.1,
                                 "interpol_factor": 2,
                                 "torque_driven": False,
                                 "use_torque": False,
                                 "save_results": True,
                                 "track_emg": True,
                                 "init_w_kalman": False,
                                 "kin_data_to_track": "markers",
                                 "mvc_list": mvc_list,
                                 "exp_freq": 32,
                                 "muscle_track_idx": [
                                     14, 25, 26,  # PEC
                                     13,  # DA
                                     15,  # DM
                                     21,  # DP
                                     23, 24,  # bic
                                     28, 29, 30,  # tri
                                     10,  # TRAPsup
                                     2,  # TRAPmed
                                     3,  # TRAPinf
                                     27,  # Lat
                                 ],
                                 "result_dir": result_dir,
                                 "result_file_name": file_name,
                                 "solver_options": solver_options,
                                 "weights": configure_weights()
                                 }

            variables_dic = {"print_lvl": 1}

            # data_to_show = ["force", "q"]
            data_to_show = None
            server_ip = "127.0.0.1"
            server_port = 50000
            MHE = MuscleForceEstimator(configuration_dic)
            MHE.run(
                variables_dic,
                server_ip,
                server_port,
                data_to_show,
                test_offline=True,
                offline_file=offline_path,
            )
