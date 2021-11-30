from .utils import *
from time import time, sleep
import biorbd_casadi as biorbd
import numpy as np
import scipy.io as sio
from casadi import MX, Function
from bioptim import (
    MovingHorizonEstimator,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    InterpolationType,
    Solver,
    Bounds,
    Node,
    OptimalControlProgram,
)


def muscle_forces(q, qdot, act, controls, model, nlp, use_excitation=False):
    muscles_states = nlp.model.stateSet()
    for k in range(model.nbMuscles()):
        if use_excitation is not True:
            muscles_states[k].setActivation(controls[k])
        else:
            muscles_states[k].setExcitation(controls[k])
            muscles_states[k].setActivation(act[k])
    muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx()
    return muscles_force


# Return biorbd muscles force function
def force_func(biorbd_model, nlp, use_excitation=False):
    qMX = MX.sym("qMX", biorbd_model.nbQ(), 1)
    dqMX = MX.sym("dqMX", biorbd_model.nbQ(), 1)
    aMX = MX.sym("aMX", biorbd_model.nbMuscles(), 1)
    uMX = MX.sym("uMX", biorbd_model.nbMuscles(), 1)
    return Function(
        "MuscleForce",
        [qMX, dqMX, aMX, uMX],
        [muscle_forces(qMX, dqMX, aMX, uMX, biorbd_model, nlp=nlp, use_excitation=use_excitation)],
        ["qMX", "dqMX", "aMX", "uMX"],
        ["Force"],
    ).expand()


def get_reference_data(file_path):
    mat = sio.loadmat(file_path)
    return mat['kalman'], mat["markers"], mat["emg"]
    # with open(file_path, "rb") as file:
    #     data = pickle.load(file)
    # states = data["data"][0]
    # controls = data["data"][1]
    # return states["q"][:, :], states["qdot"][:, :], states["muscles"][:, :], controls["muscles"][:, :]


# New function for bioptim.mhe
def define_objective(
    weights, use_excitation, use_torque, track_emg, muscles_target, kin_target, biorbd_model, kin_data_to_track="markers",
        muscle_track_idx=()
):
    # muscle_track_idx = range(0, 34)
    muscle_min_idx = []
    muscle_target_reduce = np.zeros((len(muscle_track_idx), muscles_target.shape[1]))
    count = 0
    for i in range(biorbd_model.nbMuscles()):
        if i in muscle_track_idx:
            muscle_target_reduce[count, :] = muscles_target[i, :]
            count += 1
        else:
            muscle_min_idx.append(i)

    objectives = ObjectiveList()
    if track_emg:
        # muscles_target =
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_CONTROL,
            weight=weights["track_emg"],
            target=muscle_target_reduce,
            index=muscle_track_idx,
            key="muscles",
            multi_thread=False,
        )

        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weights["min_control"], index=muscle_min_idx, key="muscles", multi_thread=False
        )
    else:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weights["min_control"], key="muscles", multi_thread=False
        )

    if use_torque:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weights["min_torque"], key="tau", multi_thread=False
        )

    kin_funct = ObjectiveFcn.Lagrange.TRACK_STATE if kin_data_to_track == "q" else ObjectiveFcn.Lagrange.TRACK_MARKERS
    kin_weight = weights["track_markers"] if kin_data_to_track == "markers" else weights["track_q"]
    if kin_data_to_track == 'markers':
        objectives.add(kin_funct, weight=kin_weight, target=kin_target, node=Node.ALL, multi_thread=False)
    elif kin_data_to_track == 'q':
        objectives.add(kin_funct, weight=kin_weight, target=kin_target, key='q', node=Node.ALL, multi_thread=False)

    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weights["min_q"],
        index=np.array(range(biorbd_model.nbQ())),
        key="q",
        node=Node.ALL,
        multi_thread=False,
    )
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weights["min_dq"],
        index=np.array(range(biorbd_model.nbQ())),
        key="qdot",
        node=Node.ALL,
        multi_thread=False,
    )
    if use_excitation is True:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            weight=weights["min_act"],
            index=np.array(range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles())),
            key="muscles",
        )
    return objectives


def prepare_mhe(
    biorbd_model, objectives, window_len, window_duration, x0, use_torque=False, use_excitation=False, nb_threads=8
):
    # Model path
    biorbd_model = biorbd_model
    nbGT = biorbd_model.nbGeneralizedTorque() if use_torque else 0
    nbMT = biorbd_model.nbMuscleTotal()
    tau_min, tau_max, tau_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.1
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.MUSCLE_DRIVEN, with_excitations=use_excitation, with_torque=use_torque, expand=False
    )

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    if use_excitation is True:
        x_bounds[0].concatenate(
            Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
        )

    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * nbGT + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [tau_max] * nbGT + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    # Initial guesses
    x_init = InitialGuess(np.concatenate((x0[:, :window_len+1], np.zeros((x0.shape[0], window_len+1)))),
                          interpolation=InterpolationType.EACH_FRAME)

    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)
    u_init = InitialGuess(np.tile(u0, (window_len, 1)).T, interpolation=InterpolationType.EACH_FRAME)

    mhe = CustomMhe(
        biorbd_model=biorbd_model,
        dynamics=dynamics,
        window_len=window_len,
        window_duration=window_duration,
        objective_functions=objectives,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=nb_threads,
    )
    # mhe = MovingHorizonEstimator(
    #     biorbd_model=biorbd_model,
    #     dynamics=dynamics,
    #     window_len=window_len,
    #     window_duration=window_duration,
    #     objective_functions=objectives,
    #     x_init=x_init,
    #     u_init=u_init,
    #     x_bounds=x_bounds,
    #     u_bounds=u_bounds,
    #     n_threads=nb_threads,
    #     # use_sx=False
    # )

    solver = Solver.ACADOS()
    solver.set_convergence_tolerance(1e-3)
    solver.set_integrator_type("IRK")
    solver.set_qp_solver("PARTIAL_CONDENSING_HPIPM")
    # solver.set_qp_solver("FULL_CONDENSING_QPOASES")
    solver.set_nlp_solver_type("SQP_RTI")
    # solver.set_nlp_solver_type("SQP")
    solver.set_sim_method_num_steps(1)
    solver.set_print_level(0)
    solver.set_maximum_iterations(10)
    # solver = Solver.IPOPT()
    # ------------- #
    return mhe, solver


def get_target(mhe, t, x_ref, markers_ref, muscles_ref, ns_mhe, markers_ratio, emg_ratio, slide_size, track_emg, kin_data_to_track, model, muscle_track_idx, offline):
    nbMT, nbQ = model.nbMuscles(), model.nbQ(),
    muscles_ref = muscles_ref if track_emg is True else np.zeros((nbMT, ns_mhe))
    q_target_idx, markers_target_idx, muscles_target_idx = [], [], []
    offline_emg_ratio = emg_ratio

    # Find objective function idx for targets
    for i in range(len(mhe.nlp[0].J)):
        if mhe.nlp[0].J[i].name == "MINIMIZE_CONTROL" and mhe.nlp[0].J[i].target is not None:
            muscles_target_idx = i
        elif mhe.nlp[0].J[i].name == "MINIMIZE_MARKERS" and mhe.nlp[0].J[i].target is not None:
            markers_target_idx = i
        elif mhe.nlp[0].J[i].name == "MINIMIZE_STATE" and mhe.nlp[0].J[i].target is not None:
            q_target_idx = i
    kin_target_idx = q_target_idx if kin_data_to_track == "q" else markers_target_idx

    # Define target
    # EMG target:
    muscle_target_reduce = np.zeros((len(muscle_track_idx), muscles_ref.shape[1]))
    count = 0
    for i in range(muscles_ref.shape[0]):
        if i in muscle_track_idx:
            muscle_target_reduce[count, :] = muscles_ref[i, :]
            count += 1
    muscle_target = muscle_target_reduce[:, ::offline_emg_ratio][:, slide_size*t: (ns_mhe + 1 + slide_size*t)] if offline else muscle_target_reduce

    # Markers target:
    markers_target = markers_ref[:3, :, ::markers_ratio][:3, :, slide_size*t: (ns_mhe + 1 + slide_size*t)] if offline else markers_ref
    for i in range(markers_target.shape[2]):
        for j in range(markers_target.shape[1]):
            if np.product(markers_target[:3, j, i]) == 0:
                markers_target[:3, j, i] = markers_ref[:3, j, i - 1]

    # Angle target:
    q_target = x_ref[:, ::markers_ratio][:, slide_size*t: (ns_mhe + 1 + slide_size*t)] if offline else x_ref[:, :]
    q_target = np.concatenate((q_target, np.zeros((nbQ, q_target.shape[1]))), axis=0) if x_ref.shape[0] < nbQ else q_target

    kin_target = q_target if kin_data_to_track == 'q' else markers_target
    target = {"kin_target": [kin_target_idx, kin_target]}
    if track_emg:
        target["muscle_target"] = [muscles_target_idx, muscle_target]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(muscle_target_reduce[:, ::offline_emg_ratio].T)
    # plt.show()
    return target


def update_mhe(mhe, t, sol, estimator_instance, muscle_track_idx, initial_time, offline_data=None):
    tic = time()
    if sol:
        print(sol.status)
    if estimator_instance.test_offline:
        x_ref, markers_ref, muscles_ref = offline_data
    else:
        # data = estimator_instance.get_data(t)
        # x_ref = np.array(data["kalman"])
        # markers_ref = np.array(data["markers"])
        # muscles_ref = np.array(data["emg"])
        while True:
            try:
                data = estimator_instance.data_queue.get_nowait()
                x_ref = np.array(data["kalman"])
                markers_ref = np.array(data["markers"])
                muscles_ref = np.array(data["emg"])
                break
            except:
                pass
    target = get_target(mhe,
                        t,
                        x_ref,
                        markers_ref,
                        muscles_ref,
                        estimator_instance.ns_mhe,
                        estimator_instance.markers_ratio,
                        estimator_instance.EMG_ratio,
                        estimator_instance.slide_size,
                        estimator_instance.track_emg,
                        estimator_instance.kin_data_to_track,
                        estimator_instance.model,
                        muscle_track_idx,
                        estimator_instance.test_offline
                        )
    mhe.slide_size = estimator_instance.slide_size
    if estimator_instance.test_offline:
        mhe.x_ref = x_ref[:, ::estimator_instance.markers_ratio][:, estimator_instance.slide_size*t:(estimator_instance.ns_mhe + 1 + estimator_instance.slide_size*t)]
        mhe.muscles_ref = muscles_ref[:, ::estimator_instance.EMG_ratio][:, estimator_instance.slide_size*t: estimator_instance.ns_mhe + estimator_instance.slide_size*t]
    else:
        mhe.x_ref = x_ref
        mhe.muscles_ref = muscles_ref
    # if t != 0:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     # plt.plot(muscles_ref.T)
    #     plt.plot(mhe.muscles_ref.T)
    #     plt.show()

    for key in target.keys():
        mhe.update_objectives_target(target=target[key][1], list_index=target[key][0])
    if t != 0:
        stat = t if sol.status != 0 else -1
    else:
        stat = -1

    if t > 0:
        force_est, q_est, dq_est, a_est, u_est = compute_force(
            sol, estimator_instance.get_force, estimator_instance.nbMT, estimator_instance.use_excitation
        )
        # import matplotlib.pyplot as plt
        # plt.plot(sol.states["q"][0, :], 'o')
        # plt.plot(target["kin_target"][1][0, :], 'x')
        # plt.plot(x_ref[:, ::estimator_instance.rt_ratio][0, t:estimator_instance.ns_mhe + t + 1])
        # plt.show()
        if estimator_instance.data_to_show:
            dic_to_put = {"t": t, "force_est": force_est, "q_est": q_est}
            try:
                estimator_instance.plot_queue.get_nowait()
            except:
                pass
            estimator_instance.plot_queue.put_nowait(dic_to_put)

        time_to_get_data = time() - tic
        time_to_solve = sol.real_time_to_optimize
        time_tot = time_to_solve + time_to_get_data

        if estimator_instance.save_results:
            data_to_save = {
                "time": time() - initial_time,
                "X_est": np.concatenate((q_est, dq_est), axis=0),
                "U_est": u_est,
                "kalman": mhe.x_ref[:, -2:-1],
                "f_est": force_est,
                "init_w_kalman": estimator_instance.init_w_kalman,
                "none_conv_iter": stat,
            }
            if t == 1:
                data_to_save["Nmhe"] = estimator_instance.ns_mhe
            data_to_save["muscles_target"] = target["muscle_target"][1][:, -1:] if "muscle_target" in target.keys() else 0
            kin_target = target["kin_target"][1][:, :, -2:-1] if estimator_instance.kin_data_to_track == "markers" else target["kin_target"][1][:, -2:-1]
            data_to_save["kin_target"] = kin_target
            # time_to_get_data = time() - tic
            # time_to_solve = sol.real_time_to_optimize
            # print(sol.real_time_to_optimize)
            # print(time_to_get_data)
            # time_tot = time_to_solve + time_to_get_data
            # print(time_tot
            #       )
            data_to_save["sol_freq"] = 1 / time_tot
            data_to_save["exp_freq"] = estimator_instance.exp_freq
            data_to_save["sleep_time"] = (1 / estimator_instance.exp_freq) - time_tot
            save_results(sol, data_to_save, estimator_instance.current_time, estimator_instance.kin_data_to_track, estimator_instance.track_emg, estimator_instance.use_torque, estimator_instance.use_excitation)

            if estimator_instance.print_lvl == 1:
                print(
                    f"Solve Frequency : {1 / time_tot} \n"
                    f"Expected Frequency : {estimator_instance.exp_freq}\n"
                    f"time to sleep: {(1 / estimator_instance.exp_freq) - time_tot}\n"
                    f"time to get data = {time_to_get_data}"
                )
        current_time = time() - tic
        time_tot = time_to_solve + current_time
        if 1 / time_tot > estimator_instance.exp_freq:
            sleep((1 / estimator_instance.exp_freq) - time_tot)
    if estimator_instance.test_offline:
        if t == 200:
            return False
        else:
            return True
    else:
        return True


class CustomMhe(MovingHorizonEstimator):
    def __init__(self, **kwargs):

        self.init_w_kalman = False
        self.x_ref = []
        self.muscles_ref = []
        self.slide_size = 0
        # self.muscles_target = []
        # self.kin_target = []

        super(CustomMhe, self).__init__(**kwargs)

    def advance_window(self, sol, steps: int = 0, **advance_options):
        x = sol.states["all"]
        u = sol.controls["all"][:, :-1]
        if self.init_w_kalman:
            x0 = np.hstack((x[:, self.slide_size:], self.x_ref[:, -self.slide_size:]))
            # x0 = self.x_ref[:, :]
        else:
            x0 = np.hstack((x[:, self.slide_size:], x[:, -self.slide_size:]))  # discard oldest estimate of the window, duplicates youngest
        u0 = np.hstack((u[:, self.slide_size:], u[:, -self.slide_size:]))
        # print(u.shape)
        # print(x.shape)
        if sol.status != 0 and sol.status != 2:
            x0[:int(x.shape[0]/2), :] = self.x_ref[:int(x.shape[0]/2), -x0.shape[1]:]
            u0[int(x.shape[0]/2):, :] = self.muscles_ref[:, -u0.shape[1]:]

        # u0 = self.muscles_ref
        x_init = InitialGuess(x0, interpolation=InterpolationType.EACH_FRAME)
        u_init = InitialGuess(u0, interpolation=InterpolationType.EACH_FRAME)
        self.update_initial_guess(x_init, u_init)


def prepare_short_ocp(model_path: str, final_time: float, n_shooting: int):
    """
    Prepare to build a blank short ocp to use single shooting bioptim function
    Parameters
    ----------
    model_path: str
        path to bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    Returns
    -------
    The blank OptimalControlProgram
    """
    biorbd_model = biorbd.Model(model_path)

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_torque=True, expand=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [0] * biorbd_model.nbGeneralizedTorque() + [0] * biorbd_model.nbMuscleTotal(),
        [10] * biorbd_model.nbGeneralizedTorque() + [1] * biorbd_model.nbMuscleTotal(),
    )

    x_init = [0] * biorbd_model.nbQ() * 2
    x_init = InitialGuess(x_init)
    u_init = InitialGuess([0] * biorbd_model.nbMuscles() + [0] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        use_sx=True,
    )
