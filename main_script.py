from biosiglive.client import Client
import multiprocessing as mp

try:
    from optim_funct import force_func, prepare_ocp, prepare_mhe, define_objective, get_solver_options
except ModuleNotFoundError:
    pass
from biosiglive.data_processing import read_data, add_data_to_pickle
from time import sleep, strftime, time
from biosiglive.data_plot import init_plot_force, init_plot_q, update_plot_force, update_plot_q
import numpy as np
import biorbd_casadi as biorbd
import scipy.io as sio
from bioptim import (
    Bounds,
    InterpolationType,
    Solver,
    InitialGuess,
    MovingHorizonEstimator,
    Solution,
    Dynamics,
    DynamicsList,
)

# TODO add class for configurate problem instead of dic
# class ConfPlot:


class MuscleForceEstimator:
    def __init__(self, *args):
        conf = self._check_and_adjust_dim(*args)
        self.model_path = conf["model_path"]  # arm26_6mark_EKF.bioMod"
        biorbd_model = biorbd.Model(self.model_path)
        self.use_torque = False
        self.use_excitation = False
        self.save_results = True
        self.TRACK_EMG = False
        self.use_N_elec = False
        self.data_to_show = [""]
        self.kin_data_to_track = None
        self.init_w_kalman = False

        # Variables of the problem
        self.exp_freq = 30
        self.Ns_mhe = 7
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

        for key in conf.keys():
            self.__dict__[key] = conf[key]

        self.T_mhe = 1 / self.exp_freq * self.Ns_mhe  # Compute the new time of OCP
        self.nbQ, self.nbMT = biorbd_model.nbQ(), biorbd_model.nbMuscles()
        self.nbGT = biorbd_model.nbGeneralizedTorque() if self.use_torque else 0
        self.current_time = strftime("%Y%m%d-%H%M")

        # def _init_mhe(self):

        self.data_to_get = []
        # if self.kin_data_to_track == "markers":
        self.data_to_get.append("markers")
        # elif self.kin_data_to_track == "q":
        #     self.data_to_get.append("q")
        # if self.TRACK_EMG:
        self.data_to_get.append("emg")

        self.muscles_target = np.zeros((self.nbMT, self.Ns_mhe))
        self.markers_target = np.zeros((3, biorbd_model.nbMarkers(), self.Ns_mhe + 1))
        self.x_ref = np.zeros((self.nbQ * 2, self.Ns_mhe + 1))
        X_est = np.ndarray((self.nbQ * 2 + self.nbMT, 1)) if self.use_excitation else np.ndarray((self.nbQ * 2, 1))
        self.kin_target = self.markers_target if self.kin_data_to_track == "markers" else self.x_ref[: self.nbQ, :]
        self.markers_ratio = int(self.markers_rate / self.exp_freq)
        self.EMG_ratio = int(self.emg_rate * self.markers_ratio / self.markers_rate)
        self.muscle_names = []
        for i in range(biorbd_model.nbMuscles()):
            self.muscle_names.append(biorbd_model.muscleNames()[i].to_string())
        self.dof_names = []
        for i in range(biorbd_model.nbQ()):
            self.dof_names.append(biorbd_model.nameDof()[i].to_string())

        # define some variables
        self.var, self.server_ip, self.server_port, self.data_to_show = {}, [], [], []

        # multiprossess stuffs
        manager = mp.Manager()
        self.data_count = mp.Value('i', 0)
        self.plot_queue = manager.Queue()
        self.data_queue = manager.Queue()
        self.data_event = mp.Event()
        self.process = mp.Process

        objectives = define_objective(
            use_excitation=self.use_excitation,
            use_torque=self.use_torque,
            TRACK_EMG=self.TRACK_EMG,
            muscles_target=self.muscles_target,
            kin_target=self.kin_target,
            biorbd_model=biorbd_model,
            kin_data_to_track=self.kin_data_to_track,
        )

        # Build the graph
        self.mhe, self.solver = prepare_mhe(
            biorbd_model,
            objectives,
            window_len=self.Ns_mhe,
            window_duration=self.T_mhe,
            x0=X_est[:, 0],
            use_torque=self.use_torque,
            use_excitation=self.use_excitation,
            nb_threads=4,
        )

        nlp = self.mhe.nlp[0]
        self.get_force = force_func(biorbd_model, nlp, use_excitation=self.use_excitation)
        self.force_est = np.ndarray((biorbd_model.nbMuscles(), 1))

    @staticmethod
    def _check_and_adjust_dim(*args):
        if len(args) == 1:
            conf = args[0]
        else:
            conf = {}
            for i in range(len(args)):
                for key in args[i].keys():
                    conf[key] = args[i][key]
        return conf

    # @staticmethod
    def _update_plot(self, t, force_est, q_est):
        # conf = MuscleForceEstimator._check_and_adjust_dim(*args)
        if self.data_to_show.count("force") != 0:
            self.force_to_plot = np.append(self.force_to_plot[:, -self.exp_freq - 1:], force_est, axis=1)
            # if self.count_p_f == self.plot_force_ratio:
            update_plot_force(
                self.force_to_plot, self.p_force, self.app_force, self.plot_force_ratio, self.muscle_names
            )  # , box_force)
            self.count_p_f = 0
            # else:
            self.count_p_f += 1

        if self.data_to_show.count("q") != 0:
            self.q_to_plot = np.append(self.q_to_plot[:, -self.exp_freq - 1 :], q_est.reshape(-1, 1), axis=1)
            # # if self.count_p_q == self.plot_force_ratio:
            # update_plot_q(self.q_to_plot * (180 / np.pi), self.p_q, self.app_q, self.box_q)
            # self.count_p_q = 0
            # self.b.load_experimental_markers(self.kin_target[:, :, -1:])
            self.b.set_q(q_est[:, -1])
            # self.b.update()
            # else:
            #     self.count_p_q += 1

    def _update_target(self, t, mhe, data):
        kin_data_to_track = self.kin_data_to_track
        nbMT, nbQ, Ns_mhe = self.nbMT, self.nbQ, self.Ns_mhe
        x_ref_tmp = np.array(data["kalman"])
        x_ref_tmp = np.append(x_ref_tmp, np.zeros((x_ref_tmp.shape[0], x_ref_tmp.shape[1])), axis=0)

        muscles_target_tmp = np.array(data["emg"])[:, :-1] #if TRACK_EMG is True else np.zeros((nbMT, Ns_mhe))
        # muscles_target = muscles_target_tmp
        # muscles_target = np.ndarray((nbMT, Ns_mhe))
        # muscles_target[[1, 2], :] = muscles_target_tmp[0, :]
        # muscles_target[0, :] = muscles_target_tmp[1, :]
        # muscles_target[[3, 4], :] = muscles_target_tmp[1, :]
        # muscles_target[5, :] = np.zeros((1, muscles_target_tmp.shape[1]))

        if kin_data_to_track == "markers":
            kin_target_tmp = np.array(data["markers"])[:3, :, :]
            # print(kin_target_tmp)
        elif kin_data_to_track == "q":
            kin_target_tmp = x_ref_tmp[:nbQ, :]
        else:
            raise RuntimeError(
                f"Not valid value for kin data to track ({kin_data_to_track}). Values are 'markers' or 'q'."
            )

        muscle_track_idx = [0, 1, 2, 3, 4, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26]
        muscles_target_reduce = np.zeros((len(muscle_track_idx), muscles_target_tmp.shape[1]))
        count = 0
        for i in range(muscles_target_tmp.shape[0]):
            if i in muscle_track_idx:
                muscles_target_reduce[count, :] = muscles_target_tmp[i, :]
                count += 1

        if t == 0:
            kin_target = kin_target_tmp
            muscles_target = muscles_target_reduce
            x_ref = x_ref_tmp

        elif t > 0:
            # if TRACK_EMG:
            muscles_target = np.append(mhe.muscles_target[:, 1:], muscles_target_reduce[:, -1:], axis=1)
            x_ref = np.append(mhe.x_ref[:, 1:], x_ref_tmp[:, -1:], axis=1)
            if kin_data_to_track == "markers":
                kin_target = np.append(mhe.kin_target[:, :, 1:], kin_target_tmp[:, :, -1:], axis=2)
            elif kin_data_to_track == "q":
                kin_target = np.append(mhe.kin_target[:, 1:], kin_target_tmp[:, -1:], axis=1)

        q_target_idx, markers_target_idx, muscles_target_idx = [], [], []
        for i in range(len(mhe.nlp[0].J)):
            if mhe.nlp[0].J[i].name == "MINIMIZE_CONTROL" and mhe.nlp[0].J[i].target is not None:
                muscles_target_idx = i
            elif mhe.nlp[0].J[i].name == "MINIMIZE_MARKERS" and mhe.nlp[0].J[i].target is not None:
                markers_target_idx = i
            elif mhe.nlp[0].J[i].name == "MINIMIZE_STATE" and mhe.nlp[0].J[i].target is not None:
                q_target_idx = i
        kin_target_idx = q_target_idx if self.kin_data_to_track == "q" else markers_target_idx

        if self.TRACK_EMG is True:
            target = {str(muscles_target_idx): muscles_target, str(kin_target_idx): kin_target}
            # target = {str(muscles_target_idx): muscles_target_tmp, str(kin_target_idx): kin_target}
        else:
            target = {str(kin_target_idx): kin_target}
            # muscles_target_reduce = muscles_target

        for key in target.keys():
            mhe.update_objectives_target(target=target[key], list_index=int(key))
        return muscles_target_reduce, kin_target, x_ref

    # @staticmethod
    def get_data(self, nb_of_data):
        data_to_get = self.data_to_get
        # nb_of_data = self.Ns_mhe + 1
        # while True:
        #     # if self.data_count.value == 1:
        #     try:
        #         # print(self.data_count.value)
        #         self.data_queue.get_nowait()
        #     except:
        #         pass
        #         # self.data_count.value = 0

        vicon_client = Client(self.server_ip, self.server_port, type="TCP")
        data = vicon_client.get_data(
            data_to_get,
            read_frequency=self.exp_freq,
            nb_of_data_to_export=nb_of_data,
            nb_frame_of_interest=self.Ns_mhe,
            get_names=self.get_names,
            get_kalman=self.get_kalman,
        )
            # self.data_queue.put_nowait(data)
            # self.data_count.value += 1
            # if self.data_event.is_set() is not True:
            #     # print(self.data_event.is_set())
            #     self.data_event.set()

        return data

    # @staticmethod
    def _update_offline_target(self, mhe, t, mat):
        # conf = MuscleForceEstimator._check_and_adjust_dim(*args)
        # TRACK_EMG = self.TRACK_EMG
        # kin_data_to_track = conf["kin_data_to_track"]
        # nbMT, nbQ, Ns_mhe = conf["nbMT"], conf["nbQ"], conf["Ns_mhe"]

        # load data if offline
        # print(mat.keys())
        markers_target = np.nan_to_num(mat["markers"][:3, :, ::self.markers_ratio])

        muscles_target_tmp = mat["emg"][:, ::self.markers_ratio] if self.TRACK_EMG is True else np.zeros((self.nbMT, self.Ns_mhe))
        x_ref = mat["kalman"][:, ::self.markers_ratio]
        # kal = sio.loadmat("kalman_test.mat")["kalman"]
        # kal = np.append(kal, np.zeros((kal.shape[0], kal.shape[1])), axis=0)
        # mat["kalman"] = kal
        q_target_idx, markers_target_idx, muscles_target_idx = [], [], []

        # x_ref = kal
        for i in range(len(mhe.nlp[0].J)):
            if mhe.nlp[0].J[i].name == "MINIMIZE_CONTROL" and mhe.nlp[0].J[i].target is not None:
                muscles_target_idx = i
            elif mhe.nlp[0].J[i].name == "MINIMIZE_MARKERS" and mhe.nlp[0].J[i].target is not None:
                markers_target_idx = i
            elif mhe.nlp[0].J[i].name == "MINIMIZE_STATE" and mhe.nlp[0].J[i].target is not None:
                q_target_idx = i
        kin_target_idx = q_target_idx if self.kin_data_to_track == "q" else markers_target_idx
        if t == 0:
            if self.kin_data_to_track == 'markers':
                kin_target = markers_target[:, :, :self.Ns_mhe + 1]
                for i in range(kin_target.shape[2]):
                    for j in range(kin_target.shape[1]):
                        if np.product(kin_target[:, j, i]) == 0:
                            kin_target[:, j, i] = markers_target[:, j, i - 1]
            else:
                kin_target = x_ref[:self.nbQ, :self.Ns_mhe + 1]
            muscles_target_tmp = muscles_target_tmp[:, :self.Ns_mhe]
            x_ref = x_ref[:, :self.Ns_mhe + 1]

        else:
            if self.kin_data_to_track == 'markers':
                kin_target = markers_target[:, :, t:self.Ns_mhe + 1 + t]
                for i in range(kin_target.shape[2]):
                    for j in range(kin_target.shape[1]):
                        if np.product(kin_target[:, j, i]) == 0:
                            kin_target[:, j, i] = markers_target[:, j, i - 1]
            else :
                # kin_target = kin_target[:self.nbQ, t:self.Ns_mhe + 1 + t]
                kin_target = x_ref[:self.nbQ, t:self.Ns_mhe + 1 + t]
            x_ref = x_ref[:, t:self.Ns_mhe + 1 + t]
            muscles_target_tmp = muscles_target_tmp[:, t:self.Ns_mhe + t] if self.TRACK_EMG is True else np.zeros((self.nbMT, self.Ns_mhe))

        # kin_target = x_ref[:self.nbQ, :] if self.kin_data_to_track == "q" else markers_target
        if self.TRACK_EMG is True:
            # muscles_target = np.ndarray((self.nbMT, self.Ns_mhe))
            # muscles_target[[1, 2], :] = muscles_target_tmp[0, :]
            # muscles_target[[0, 3, 4], :] = muscles_target_tmp[1, :]
            # muscles_target[5, :] = np.zeros((1, muscles_target_tmp.shape[1]))
            muscle_track_idx = [0, 1, 2, 3, 4, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26]
            muscle_target_reduce = np.zeros((len(muscle_track_idx), muscles_target_tmp.shape[1]))
            count = 0
            for i in range(muscles_target_tmp.shape[0]):
                if i in muscle_track_idx:
                    muscle_target_reduce[count, :] = muscles_target_tmp[i, :]
                    count += 1
            target = {str(muscles_target_idx): muscle_target_reduce, str(kin_target_idx): kin_target}
        else:
            target = {str(kin_target_idx): kin_target}
            muscle_target_reduce = muscles_target_tmp

        for key in target.keys():
            mhe.update_objectives_target(target=target[key], list_index=int(key))
        return muscle_target_reduce, kin_target, x_ref

    @staticmethod
    def _compute_force(sol, get_force, nbMT, use_excitation=False):
        force_est = np.zeros((nbMT, 1))
        q_est = sol.states["q"][:, -2:-1]
        dq_est = sol.states["qdot"][:, -2:-1]
        if use_excitation:
            a_est = sol.states["muscles"][:, -1:]
            u_est = sol.controls["muscles"][:, -2:-1]
        else:
            a_est = sol.controls["muscles"][:, -2:-1]
            u_est = a_est
        # Compute force
        for i in range(nbMT):
            force_est[i, 0] = get_force(q_est, dq_est, a_est, u_est)[i, :]
        return force_est, q_est, dq_est, a_est, u_est

    # @staticmethod
    def _save_results(self, sol, data_to_save):
        current_time = self.current_time
        if self.use_torque:
            data_to_save["tau_est"] = sol.controls["tau"][:, -2:-1]

        dyn = "act" if self.use_excitation is not True else "exc"
        torque = "_torque" if self.use_torque else ""
        EMG = "EMG_" if self.TRACK_EMG else ""
        data_path = f"Solution/test_wu_model/Results_MHE_{self.kin_data_to_track}_{EMG}{dyn}{torque}_driven_test_{current_time}"
        add_data_to_pickle(data_to_save, data_path)

    def _warm_start(self, sol):
        if self.mhe.nlp[0].x_bounds.type != InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if self.mhe.nlp[0].x_bounds.type == InterpolationType.CONSTANT:
                x_min = np.repeat(self.mhe.nlp[0].x_bounds.min[:, :1], 3, axis=1)
                x_max = np.repeat(self.mhe.nlp[0].x_bounds.max[:, :1], 3, axis=1)
                self.mhe.nlp[0].x_bounds = Bounds(x_min, x_max)
            else:
                raise NotImplementedError(
                    "The MHE is not implemented yet for x_bounds not being "
                    "CONSTANT or CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT"
                )
            self.mhe.nlp[0].x_bounds.check_and_adjust_dimensions(self.mhe.nlp[0].states.shape, 3)
        self.mhe.nlp[0].x_bounds[:, 0] = sol.states["all"][:, 1]
        x = sol.states["all"]
        u = sol.controls["all"][:, :-1]
        if self.init_w_kalman:
            x0 = np.hstack((x[:, 1:], self.x_ref[:, -1:]))
        else:
            x0 = np.hstack((x[:, 1:], x[:, -1:]))  # discard oldest estimate of the window, duplicates youngest
        u0 = np.hstack((u[:, 1:], u[:, -1:]))
        x_init = InitialGuess(x0, interpolation=InterpolationType.EACH_FRAME)
        u_init = InitialGuess(u0, interpolation=InterpolationType.EACH_FRAME)
        self.mhe.update_initial_guess(x_init, u_init)

    def run(self, var, server_ip, server_port, data_to_show=None, test_offline=False):
        self.var = var
        self.server_ip = server_ip
        self.server_port = server_port
        self.data_to_show = data_to_show
        self.test_offline = test_offline
        proc_plot, proc_get_data = [], []
        # if test_offline is not True:
        #     proc_get_data = self.process(name="data", target=MuscleForceEstimator.get_data, args=(self,))
        #     proc_get_data.start()
        # self.run_mhe(var, server_ip, server_port, data_to_show)
        proc_mhe = self.process(name="mhe", target=MuscleForceEstimator.run_mhe, args=(self, var, server_ip, server_port, data_to_show))
        proc_mhe.start()

        if self.data_to_show is not None:
            proc_plot = self.process(name="plot", target=MuscleForceEstimator.run_plot, args=(self,))
            proc_plot.start()

        proc_mhe.join()
        # if test_offline is not True:
        #     proc_get_data.join()

        if self.data_to_show is not None:
            proc_plot.join()

    def run_plot(self):
        # print(1)
        # self.data_to_show = data_to_show
        # initialize plot
        # if data_to_show:
        for data in self.data_to_show:
            if data == "force":
                # p_force, win_force, app_force, box_force = init_plot_force(self.nbMT)
                self.p_force, self.win_force, self.app_force = init_plot_force(self.nbMT)
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

        while True:
            try:
                data = self.plot_queue.get_nowait()
                is_working = True
            except:
                is_working = False
            if is_working:
                self._update_plot(data["t"], data["force_est"], data["q_est"])

    def run_mhe(self, var, server_ip, server_port, data_to_show):
        # var, server_ip, server_port = self.var, self.server_ip, self.server_port
        self.data_to_show = data_to_show
        for key in var.keys():
            if key in self.__dict__:
                self.__setattr__(key, var[key])
            else:
                raise RuntimeError(f'{key} is not a variable of the class')

        # if "N_elec" in keys is not True:
        # N_elec = 2 if "N_elec" in keys is not True else var["N_elec"]

        if self.test_offline:
            mat = sio.loadmat(f"test_abd.mat")
        self.mhe.init_w_kalman = self.init_w_kalman
        self.mhe.x_ref = self.x_ref
        initial_time = time()
        self.mhe.kin_target = self.kin_target
        self.mhe.muscles_target = self.muscles_target

        # if data_to_show:
        #     if "q" in data_to_show:
        #         import bioviz
        #         self.b = bioviz.Viz(model_path=self.model_path,
        #                             show_global_center_of_mass=False,
        #                             show_markers=True,
        #                             show_muscles=False,
        #                             show_segments_center_of_mass=False,
        #                             show_local_ref_frame=False,
        #                             show_global_ref_frame=False
        #                             )
        #     for show_data in self.data_to_show:
        #         if show_data == "force":
        #             # p_force, win_force, app_force, box_force = init_plot_force(nbMT)
        #             self.p_force, self.win_force, self.app_force = init_plot_force(self.nbMT)
        #         # elif data == "q":
        #         #     self.p_q, self.win_q, self.app_q, self.box_q = init_plot_q(self.nbQ, self.dof_names)
        #     self.plot_force_ratio = int(self.exp_freq / self.plot_force_freq)
        #     self.plot_q_ratio = int(self.exp_freq / self.plot_q_freq)
        #     self.force_to_plot = np.zeros((self.nbMT, self.plot_force_ratio))
        #     self.q_to_plot = np.zeros((self.nbQ, self.plot_q_ratio))
        #     self.count_p_f, self.count_p_q = self.plot_force_ratio, self.plot_q_ratio

        def update_functions(mhe, t, sol):
            tic = time()
            if t == 0:
                stat = -1
                nb_of_data = self.Ns_mhe + 1
            else:
                stat = t if sol.status != 0 else -1
                nb_of_data = self.Ns_mhe + 1

            if self.test_offline:
                muscles_target, markers_target, x_ref = self._update_offline_target(mhe, t, mat)
                mhe.muscles_target, mhe.kin_target = muscles_target, markers_target
                mhe.x_ref = x_ref

            else:
                data = self.get_data(nb_of_data=nb_of_data)
                # data = self.data_queue.get()
                # self.data_count.value = 0
                muscles_target, markers_target, x_ref = self._update_target(t, mhe, data)
                mhe.muscles_target, mhe.kin_target = muscles_target, markers_target
                mhe.x_ref = x_ref

            if t > 0:
                force_est, q_est, dq_est, a_est, u_est = MuscleForceEstimator._compute_force(
                    sol, self.get_force, self.nbMT, self.use_excitation
                )
                if data_to_show:
                    dic_to_put = {"t": t, "force_est": force_est, "q_est": q_est}
                    try:
                        self.plot_queue.get_nowait()
                    except:
                        pass
                    self.plot_queue.put_nowait(dic_to_put)
                    # self._update_plot(t, force_est, q_est)

                if conf["save_results"]:
                    data_to_save = {
                        "time": time() - initial_time,
                        "X_est": np.concatenate((q_est, dq_est), axis=0),
                        "U_est": u_est,
                        "kalman": x_ref[:, -2:-1],
                        "f_est": force_est,
                        "init_w_kalman": self.init_w_kalman,
                        "none_conv_iter": stat,
                    }
                    if t == 1:
                        data_to_save["Nmhe"] = self.Ns_mhe

                    data_to_save["muscles_target"] = muscles_target[:, -1:]
                    kin_target = markers_target[:, :, -2:-1] if self.kin_data_to_track == "markers" else x_ref[:, -2:-1]
                    data_to_save["kin_target"] = kin_target
                    time_to_get_data = time() - tic
                    time_to_solve = sol.real_time_to_optimize
                    time_tot = time_to_solve + time_to_get_data
                    data_to_save["sol_freq"] = 1 / time_tot
                    data_to_save["exp_freq"] = self.exp_freq
                    data_to_save["sleep_time"] = (1 / self.exp_freq) - time_tot
                    self._save_results(sol, data_to_save)

                    if self.print_lvl == 1:
                        print(
                            f"Solve Frequency : {1 / time_to_solve} \n"
                            f"Expected Frequency : {self.exp_freq}\n"
                            f"time to sleep: {(1 / self.exp_freq) - time_tot}\n"
                            f", time to get data = {time_to_get_data}"
                        )
                time_to_get_data = time() - tic
                time_to_solve = sol.real_time_to_optimize
                time_tot = time_to_solve + time_to_get_data
                if 1 / time_tot > self.exp_freq:
                    sleep((1 / self.exp_freq) - time_tot)
            if t == 150:
                return False
            else:
                return True
        # if self.test_offline is not True:
        #     self.data_event.wait()
        final_sol = self.mhe.solve(
            update_functions,
            solver=self.solver,
            export_options={"frame_to_export": self.Ns_mhe-1},
        )

        final_sol.graphs()
        return final_sol


if __name__ == "__main__":
    # sleep(5)
    conf = {
        # "model_path": "arm26_6mark_scaled.bioMod",
        "model_path": "models/wu_model.bioMod",
        "Ns_mhe": 7,
        "exp_freq": 30,
        "use_torque": True,
        "use_excitation": False,
        "save_results": False,
        "TRACK_EMG": False,
        "init_w_kalman": True,
        "kin_data_to_track": "q",
    }
    # if conf["kin_data_to_track"] == 'markers':
    #     conf["exp_freq"] = 60
    # else:
    #     conf["exp_freq"] = 80

    var = {
        "plot_force_freq": 10,
        "emg_rate": 2000,
        "markers_rate": 100,
        "plot_q_freq": 20,
        "print_lvl": 1,
    }
    # data_to_show = ["force", "q"]
    data_to_show = None
    server_ip = "127.0.0.1"
    server_port = 50000
    # data_to_show = [""]
    MHE = MuscleForceEstimator(conf)
    # sol = MHE.run_mhe(var, server_ip, server_port, data_to_show)
    MHE.run(var, server_ip, server_port, data_to_show, test_offline=True)
    # print(sol.controls)
    # sol.animate(mesh=True)
    # sol.graphs()
