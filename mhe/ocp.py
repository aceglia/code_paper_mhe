"""
This code provide every function needed to solve the OCP problem.
"""
import bioptim
from .utils import *
from time import time, sleep
import biorbd_casadi as biorbd
from biosiglive.io.save_data import read_data
import numpy as np
import datetime
# import matplotlib.pyplot as plt

import scipy.io as sio
from scipy.interpolate import interp1d
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
    Node,
    Bounds,
    BiorbdInterface
)

def muscle_forces(
    q: np.ndarray,
    qdot: np.ndarray,
    act: np.ndarray,
    controls: np.ndarray,
    model: biorbd.Model,
    use_excitation: bool = False,
):
    """
    Compute the muscle force for a given state and controls.

    Parameters
    ----------
    q : np.ndarray
        State of the model.
    qdot : np.ndarray
        State of the model.
    act : np.ndarray
        Activation of the muscles.
    controls : np.ndarray
        Controls of the model.
    model : biorbd.Model
        Model of the system.
    use_excitation : bool
        If True, use the excitation of the muscles.

    Returns
    -------
    Muscle force.
    """
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        if use_excitation is not True:
            muscles_states[k].setActivation(controls[k])
        else:
            muscles_states[k].setExcitation(controls[k])
            muscles_states[k].setActivation(act[k])
    muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx()
    return muscles_force


def force_func(biorbd_model: biorbd.Model, use_excitation: bool = False):
    """
    Define the casadi function that compute the muscle force.

    Parameters
    ----------
    biorbd_model : biorbd.Model
        Model of the system.
    use_excitation : bool
        If True, use the excitation of the muscles.

    Returns
    -------
    Casadi function that compute the muscle force.
    """
    qMX = MX.sym("qMX", biorbd_model.nbQ(), 1)
    dqMX = MX.sym("dqMX", biorbd_model.nbQ(), 1)
    aMX = MX.sym("aMX", biorbd_model.nbMuscles(), 1)
    uMX = MX.sym("uMX", biorbd_model.nbMuscles(), 1)
    return Function(
        "MuscleForce",
        [qMX, dqMX, aMX, uMX],
        [muscle_forces(qMX, dqMX, aMX, uMX, biorbd_model, use_excitation=use_excitation)],
        ["qMX", "dqMX", "aMX", "uMX"],
        ["Force"],
    ).expand()


def get_reference_data(file_path: str):
    """
    Get the reference data from a .mat file or pickle file. Keys must be well define to be found.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    Reference data.
    """
    nfinal = 150
    if file_path[-4:] == ".mat":
        mat = sio.loadmat(file_path)
        x_ref, markers, muscles = mat["kalman"], mat["markers"], mat["emg_proc"]

    else:
        mat = read_data(file_path)
        try:
            x_ref, markers, muscles = mat["kalman"], mat["kin_target"], mat["muscles_target"]
        except:
            x_ref, markers, muscles = mat["kalman"][:, :nfinal], mat["markers"][:, :, :nfinal], mat["emg"][:, :nfinal]
    return x_ref, markers, muscles


def custom_initial_states_objective(all_pn, key, target=None, slide_size=None):
    states = all_pn.nlp.states["q"].mx
    x_ref = np.ones((9, 10))
    # mx_xref = MX.sym("xref", states.shape[0], 1)
    from casadi import DM
    x_ref_dm = DM(x_ref.shape[0], x_ref.shape[1])
    # Function("state_dif", mx_xref, )
    for i in range(x_ref.shape[0]):
        # for j in range(x_ref.shape[1]):
        x_ref_dm[i, :] = x_ref[i, :]
    dif = x_ref - states
    dif = BiorbdInterface.mx_to_cx("state_dif", dif, all_pn.nlp.states["q"])
    return dif

def custom_initial_controls_objective(all_pn, u_ref, slide_size):
    controls = all_pn.nlp.controls["all"].mx
    return u_ref.to_mx() - controls

def custom_subproblem_derivative_objective():
    pass

def define_objective(
    weights: dict,
    use_torque: bool,
    track_emg: bool,
    muscles_target: np.ndarray,
    kin_target: np.ndarray,
    biorbd_model: biorbd.Model,
    previous_sol,
    kin_data_to_track: str = "markers",
    muscle_track_idx: list = (),
):
    """
    Define the objective function of the OCP.

    Parameters
    ----------
    weights : dict
        Weights of the different terms.
    use_torque : bool
        If True, use the torque are used in the dynamics.
    track_emg : bool
        If True, track the EMG are tracked.
    muscles_target : np.ndarray
        Target of the muscles.
    kin_target : np.ndarray
        Target for kinematics objective.
    biorbd_model : biorbd.Model
        Model of the system.
    kin_data_to_track : str
        Kind of kinematics data to track ("markers" or "q").
    muscle_track_idx : list
        Index of the muscles to track.

    Returns
    -------
    Objective function.
    """
    previous_q, previous_qdot = previous_sol[:biorbd_model.nbQ(), :], previous_sol[biorbd_model.nbQ():biorbd_model.nbQ() * 2, :]
    muscle_min_idx = []
    for i in range(biorbd_model.nbMuscles()):
        if i not in muscle_track_idx:
            muscle_min_idx.append(i)

    objectives = ObjectiveList()
    if track_emg:
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_CONTROL,
            weight=weights["track_emg"],
            target=muscles_target,
            index=muscle_track_idx,
            key="muscles",
            multi_thread=False,
        )

        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            weight=weights["min_control"],
            index=muscle_min_idx,
            key="muscles",
            multi_thread=False,
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
    if kin_data_to_track == "markers":
        kin_target_thorax = kin_target[:3, :4, :]
        kin_target_bodies = kin_target[:3, 4:, :]
        idx_thorax = [0, 1, 2, 3]
        thorax_weight_ratio = 1000
        idx_bodies = list(range(4, biorbd_model.nbMarkers()))
        objectives.add(
            kin_funct,
            weight=kin_weight / thorax_weight_ratio,
            target=kin_target_thorax,
            marker_index=idx_thorax,
            node=Node.ALL,
            multi_thread=False,
        )
        # objectives.add(
        #     kin_funct,
        #     weight=kin_weight / thorax_weight_ratio,
        #     target=kin_target_thorax[:,:, -2][:, :, np.newaxis],
        #     marker_index=idx_thorax,
        #     node=Node.END,
        #     multi_thread=False,
        # )
        objectives.add(
            kin_funct,
            weight=kin_weight,
            target=kin_target_bodies,
            marker_index=idx_bodies,
            node=Node.ALL,
            multi_thread=False,
        )
        # objectives.add(
        #     kin_funct,
        #     weight=kin_weight,
        #     target=kin_target_bodies[:,:,-2][:, :, np.newaxis],
        #     marker_index=idx_bodies,
        #     node=Node.END,
        #     multi_thread=False,
        # )

    elif kin_data_to_track == "q":
        # objectives.add(kin_funct, weight=kin_weight, target=kin_target[:, 1:-2], key="q", node=Node.INTERMEDIATES, multi_thread=False)
        objectives.add(kin_funct, weight=kin_weight, target=kin_target[:, :], key="q", node=Node.ALL, multi_thread=False)

    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weights["min_q"],
        # index=np.array(range(biorbd_model.nbQ())),
        key="q",
        node=Node.ALL,
        multi_thread=False,
        # derivative=True,
    )

    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weights["min_dq"],
        # index=np.array(range(biorbd_model.nbQ())),
        key="qdot",
        node=Node.ALL,
        multi_thread=False,
    )
    # objectives.add(
    #     ObjectiveFcn.Lagrange.MINIMIZE_STATE,
    #     weight=weights["min_dq"]*10,
    #     # index=np.array(range(biorbd_model.nbQ())),
    #     key="qdot",
    #     node=Node.ALL,
    #     multi_thread=False,
    #     derivative=True,
    # )
    # objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=1, target=previous_q, key="q",
    #                node=Node.ALL, multi_thread=False)
    # objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=1, target=previous_qdot, key="qdot",
    #                node=Node.ALL, multi_thread=False)

    return objectives


def prepare_problem(
    model_path: str,
    objectives: ObjectiveList,
    window_len: int,
    window_duration: float,
    x0: np.ndarray,
    u0: np.ndarray = None,
    use_torque: bool = False,
    nb_threads: int = 8,
    solver_options: dict = None,
    use_acados: bool = False,
):
    """
    Prepare the ocp problem and the solver to use

    parameters
    -----------
    model_path : str
        Path to the model
    objectives : ObjectiveList
        List of objectives
    window_len : int
        Length of the window
    window_duration : float
        Duration of the window
    x0 : np.ndarray
        Initial state
    u0 : np.ndarray
        Initial control
    use_torque : bool
        Use torque as control
    nb_threads : int
        Number of threads to use
    solver_options : dict
        Solver options
    use_acados : bool
        Use acados solver

    Returns
    -------
    The problem and the solver.
    """
    biorbd_model = biorbd.Model(model_path)
    nbGT = biorbd_model.nbGeneralizedTorque() if use_torque else 0
    tau_min, tau_max, tau_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.01

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_excitations=False, with_torque=use_torque, expand=False)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    x_bounds[0].min[biorbd_model.nbQ():, :] = [[-5] * 3] * biorbd_model.nbQ()
    x_bounds[0].max[biorbd_model.nbQ():, :] = [[5] * 3] * biorbd_model.nbQ()
    # x_bounds[0].min[:, 0] = x0[:, 0]
    # x_bounds[0].max[:, 0] = x0[:, 0]
    # Control path constraint
    # u_bounds = BoundsList()
    # u_bounds.add(
    #     [tau_min] * nbGT + [muscle_min] * biorbd_model.nbMuscleTotal(),
    #     [tau_max] * nbGT + [muscle_max] * biorbd_model.nbMuscleTotal(),
    # interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    u_bounds = Bounds([tau_min] * nbGT + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [tau_max] * nbGT + [muscle_max] * biorbd_model.nbMuscleTotal(), interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    # Initial guesses
    if x0.shape[0] != biorbd_model.nbQ() * 2:
        x_init = InitialGuess(
            np.concatenate((x0[:, : window_len + 1], np.zeros((x0.shape[0], window_len + 1)))),
            interpolation=InterpolationType.EACH_FRAME,
        )
    else:
        x_init = InitialGuess(x0[:, : window_len + 1], interpolation=InterpolationType.EACH_FRAME)

    if u0 is None:
        u0 = np.array([tau_init] * nbGT + [muscle_init] * biorbd_model.nbMuscles())
        u_init = InitialGuess(np.tile(u0, (window_len, 1)).T, interpolation=InterpolationType.EACH_FRAME)
    else:
        u_init = InitialGuess(u0[:, :window_len], interpolation=InterpolationType.EACH_FRAME)

    problem = CustomMhe(
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
        use_sx=use_acados,
    )
    if not use_acados:
        solver = Solver.IPOPT()
        solver.set_linear_solver("ma57")
        solver.set_print_level(5)
        # solver.set_hessian_approximation("limited-memory")
        solver.set_hessian_approximation("exact")
        solver.set_maximum_iterations(50)

    else:
        solver = Solver.ACADOS()
        solver.set_convergence_tolerance([1e-5, 1e-5, 1e-5, 1e-5])
        solver.set_integrator_type("IRK")
        solver.set_qp_solver("PARTIAL_CONDENSING_OSQP")#PARTIAL_CONDENSING_HPIPMPARTIAL_CONDENSING_OSQP
        solver.set_nlp_solver_type("SQP_RTI")
        solver.set_print_level(0)
        for key in solver_options.keys():
            solver.set_option(val=solver_options[key], name=key)
        solver.set_sim_method_num_steps(1)

    return problem, solver


def configure_weights():
    """
    Configure the weights for the objective functions

    Returns
    -------
    weights : dict
        Dictionary of weights
    """
    # with acados :
    weights = {
        "track_markers": 10000000000,
        "track_q": 1000000000,
        "min_control": 10000,
        "min_dq": 10,
        "min_q": 1,
        "min_torque": 1000,
        "min_act": 1,
        "track_emg": 500000,
        "min_control_tot": 10,
    }
    # with Ipopt:
    # weights = {
    #     "track_markers": 10000,
    #     "min_control": 1,
    #     "min_dq": 0.01,
    #     "min_q": 0.01,
    #     "track_emg": 100,
    # }
    return weights


def get_target(
    mhe,
    t: float,
    x_ref: np.ndarray,
    markers_ref: np.ndarray,
    muscles_ref: np.ndarray,
    ns_mhe: int,
    slide_size: int,
    track_emg: bool,
    kin_data_to_track: str,
    model: biorbd.Model,
    offline: bool,
    sol,
):
    """
    Get the target for the next MHE problem and the objective functions index.

    Parameters
    ----------
    mhe : CustomMhe
        The MHE problem
    t : float
        The current time
    x_ref : np.ndarray
        The reference state
    markers_ref : np.ndarray
        The reference markers
    muscles_ref : np.ndarray
        The reference muscles
    ns_mhe : int
        The number of node of the MHE problem
    slide_size : int
        The size of the sliding window
    track_emg : bool
        Whether to track EMG
    kin_data_to_track : str
        The kin_data to track
    model : biorbd.Model
        The model
    offline : bool
        Whether to use offline data

    Returns
    -------
    Dictionary of targets (values and objective functions index)
    """
    nbMT, nbQ = model.nbMuscles(), model.nbQ()
    muscles_ref = muscles_ref if track_emg is True else np.zeros((nbMT, ns_mhe))
    q_target_idx, markers_target_idx, muscles_target_idx = [], [], []

    # Find objective function idx for targets
    for i in range(len(mhe.nlp[0].J)):
        if mhe.nlp[0].J[i].name == "MINIMIZE_CONTROL" and mhe.nlp[0].J[i].target is not None:
            muscles_target_idx = i
        elif mhe.nlp[0].J[i].name == "MINIMIZE_MARKERS" and mhe.nlp[0].J[i].target is not None:
            markers_target_idx.append(i)
        elif mhe.nlp[0].J[i].name == "MINIMIZE_STATE" and mhe.nlp[0].J[i].target is not None:
            q_target_idx = i
    kin_target_idx = q_target_idx if kin_data_to_track == "q" else markers_target_idx

    # Define target
    muscle_target = muscles_ref[:, slide_size * t : (ns_mhe + 1 + slide_size * t)] if offline else muscles_ref

    # Markers target:
    markers_target_thorax = (
        markers_ref[:3, :4, slide_size * t : (ns_mhe + 1 + slide_size * t)] if offline else markers_ref[:3, :4, :]
    )
    markers_target_bodies = (
        markers_ref[:3, 4:, slide_size * t : (ns_mhe + 1 + slide_size * t)] if offline else markers_ref[:3, 4:, :]
    )

    # Angle target:
    q_target = x_ref[:nbQ, slide_size * t : (ns_mhe + 1 + slide_size * t)] if offline else x_ref
    if sol:
        previous_sol = np.concatenate(
            (sol.states["all"][:, slide_size:], np.repeat(sol.states["all"][:, -1][:, np.newaxis],
                                                               slide_size, axis=1)), axis=1
        )
    else:
        previous_sol = np.concatenate((x_ref[:, : ns_mhe + 1], np.zeros((x_ref.shape[0], ns_mhe + 1))))


    # q_target = (
    #     np.concatenate((q_target, np.zeros((nbQ, q_target.shape[1]))), axis=0) if x_ref.shape[0] < nbQ else q_target
    # )

    if kin_data_to_track == "q":
        kin_target = q_target
    else:
        kin_target_thorax = markers_target_thorax
        kin_target_bodies = markers_target_bodies
        kin_target = [kin_target_thorax, kin_target_bodies]

    target = {"kin_target": [kin_target_idx, kin_target]}#, "previous_q": [q_target_idx - 1, previous_sol[:9, :]],
              # "previous_q_dot": [q_target_idx , previous_sol[9:18, :]]
              # }
    # target = {}
    if track_emg:
        target["muscle_target"] = [muscles_target_idx, muscle_target]
    return target


def update_mhe(mhe, t: int, sol: bioptim.Solution, estimator_instance, initial_time: float, offline_data: bool = None):
    """
    Update the MHE problem with the current data.

    Parameters
    ----------
    mhe : CustomMhe
        The MHE problem
    t : int
        The current time
    sol : bioptim.Solution
        The solution of the previous problem
    estimator_instance : instance of the estimator class
        The estimator instance
    initial_time : float
        The initial time
    offline_data : bool
        Whether to use offline data

    Returns
    -------
    if online : True
    else : True if there are still target available, False otherwise
    """
    tic = time()
    mhe.frame_to_export = estimator_instance.frame_to_save
    # target to save
    x_ref_to_save = []
    muscles_target_to_save = []
    kin_target_to_save = []
    if mhe.x_ref is not None:
        x_ref_to_save = mhe.x_ref
        muscles_target_to_save = mhe.muscles_ref
        kin_target_to_save = mhe.kin_target
        # plt.figure()
        # for i in range(kin_target_to_save.shape[0]):
        #     plt.subplot(3, 3, i + 1)
        #     plt.plot(x_ref_to_save[i, :])
        #     plt.plot(kin_target_to_save[i, :])
        # plt.show()

    absolute_time_frame = 0
    absolute_delay_tcp = 0
    absolute_time_received_s = 0
    absolute_time_received_dic = 0
    if estimator_instance.test_offline:
        x_ref, markers_target, muscles_target = offline_data
        absolute_time_received = datetime.datetime.now()
        absolute_time_received_dic = {
            "day": absolute_time_received.day,
            "hour": absolute_time_received.hour,
            "hour_s": absolute_time_received.hour * 3600,
            "minute": absolute_time_received.minute,
            "minute_s": absolute_time_received.minute * 60,
            "second": absolute_time_received.second,
            "millisecond": int(absolute_time_received.microsecond / 1000),
            "millisecond_s": int(absolute_time_received.microsecond / 1000) * 0.001,
        }
        absolute_time_frame_s = 0
        absolute_time_received_s = 0
        for key in absolute_time_received_dic.keys():
            if key == "second" or key[-1:] == "s":
                absolute_time_received_s = absolute_time_received_s + absolute_time_received_dic[key]
    else:
        data = estimator_instance.get_data(t)
        x_ref = np.array(data["kalman"])
        markers_target = np.array(data["markers"])
        muscles_target = np.array(data["emg"])
        absolute_time_frame = data["absolute_time_frame"]
        vicon_latency = data["vicon_latency_total"]
        absolute_time_received = datetime.datetime.now()
        absolute_time_received_dic = {
            "day": absolute_time_received.day,
            "hour": absolute_time_received.hour,
            "hour_s": absolute_time_received.hour * 3600,
            "minute": absolute_time_received.minute,
            "minute_s": absolute_time_received.minute * 60,
            "second": absolute_time_received.second,
            "millisecond": int(absolute_time_received.microsecond / 1000),
            "millisecond_s": int(absolute_time_received.microsecond / 1000) * 0.001,
        }

        absolute_time_frame_s = 0
        absolute_time_received_s = 0
        for key in absolute_time_frame.keys():
            if key == "second" or key[-1:] == "s":
                absolute_time_frame_s = absolute_time_frame_s + absolute_time_frame[key]
                absolute_time_received_s = absolute_time_received_s + absolute_time_received_dic[key]
        absolute_delay_tcp = absolute_time_received_s - absolute_time_frame_s

    x_ref, markers_ref, muscles_target = interpolate_data(
        estimator_instance.interpol_factor, x_ref, muscles_target, markers_target
    )

    muscles_ref = muscle_mapping(
        muscles_target_tmp=muscles_target,
        mvc_list=estimator_instance.mvc_list,
        muscle_track_idx=estimator_instance.muscle_track_idx,
    )
    target = get_target(
        mhe=mhe,
        t=t,
        x_ref=x_ref,
        markers_ref=markers_ref,
        muscles_ref=muscles_ref,
        ns_mhe=estimator_instance.ns_mhe,
        slide_size=estimator_instance.slide_size,
        track_emg=estimator_instance.track_emg,
        kin_data_to_track=estimator_instance.kin_data_to_track,
        model=estimator_instance.model,
        offline=estimator_instance.test_offline,
        sol=sol
    )
    mhe.slide_size = estimator_instance.slide_size
    if estimator_instance.test_offline:
        mhe.x_ref = x_ref[
            :, estimator_instance.slide_size * t : (estimator_instance.ns_mhe + 1 + estimator_instance.slide_size * t)
        ]
    else:
        mhe.x_ref = x_ref

    mhe.muscles_ref = target["muscle_target"][1]
    if estimator_instance.kin_data_to_track == "q":
        mhe.kin_target = target["kin_target"][1]
    else:
        if isinstance(target["kin_target"][1], list):
            mhe.kin_target = np.concatenate((target["kin_target"][1][0], target["kin_target"][1][1]), axis=1)
        else:
            mhe.kin_target = target["kin_target"][1]

    for key in target.keys():
        if isinstance(target[key][1], list):
            for i in range(len(target[key][1])):
                mhe.update_objectives_target(target=target[key][1][i], list_index=target[key][0][i])
        else:
            mhe.update_objectives_target(target=target[key][1], list_index=target[key][0])

    if t != 0:
        stat = t if sol.status != 0 else -1
    else:
        stat = -1

    if t > 0:
        force_est, q_est, dq_est, a_est, u_est = compute_force(
            sol, estimator_instance.get_force, estimator_instance.nbMT, frame_to_save=estimator_instance.frame_to_save
        )

        if estimator_instance.data_to_show:
            dic_to_put = {
                "t": t,
                "force_est": force_est.tolist(),
                "q_est": q_est.tolist(),
                "init_time_frame": absolute_time_received_s,
            }
            try:
                estimator_instance.plot_queue.get_nowait()
            except:
                pass
            estimator_instance.plot_queue.put_nowait(dic_to_put)

        time_to_get_data = time() - tic
        time_to_solve = sol.real_time_to_optimize
        time_tot = time_to_solve + time_to_get_data

        if estimator_instance.save_results:
            if mhe.kalman is not None:
                mhe.kalman = np.append(mhe.kalman, x_ref_to_save[:, estimator_instance.frame_to_save:estimator_instance.frame_to_save + estimator_instance.slide_size], axis=1)
            else:
                mhe.kalman = x_ref_to_save[:, estimator_instance.frame_to_save:estimator_instance.frame_to_save + estimator_instance.slide_size]

            data_to_save = {
                "time": time() - initial_time,
                "X_est": np.concatenate((q_est, dq_est), axis=0),
                "U_est": u_est,
                "kalman": x_ref_to_save[:, estimator_instance.frame_to_save:estimator_instance.frame_to_save + estimator_instance.slide_size],
                # "kalman": x_ref_to_save[:, 1:2],
                "f_est": force_est,
                "none_conv_iter": stat,
                "solver_options": estimator_instance.solver_options,
            }
            if t == 1:
                data_to_save["Nmhe"] = estimator_instance.ns_mhe
            data_to_save["muscles_target"] = muscles_target_to_save[:, estimator_instance.frame_to_save:estimator_instance.frame_to_save + estimator_instance.slide_size] if "muscle_target" in target.keys() else 0
            # data_to_save["muscles_target"] = muscles_target_to_save[:, 0:1] if "muscle_target" in target.keys() else 0

            if estimator_instance.kin_data_to_track == "q":
                kin_target = kin_target_to_save[:, estimator_instance.frame_to_save:estimator_instance.frame_to_save + estimator_instance.slide_size]
                # kin_target = kin_target_to_save[:, 1:2]

            else:
                kin_target = kin_target_to_save[:, :, estimator_instance.frame_to_save:estimator_instance.frame_to_save + estimator_instance.slide_size]
                # kin_target = kin_target_to_save[:, :, 1:2]

            if estimator_instance.use_torque:
                data_to_save["tau_est"] = sol.controls["tau"][:, estimator_instance.frame_to_save:estimator_instance.frame_to_save + estimator_instance.slide_size]

            data_to_save["kin_target"] = kin_target
            data_to_save["sol_freq"] = 1 / time_tot
            data_to_save["exp_freq"] = estimator_instance.exp_freq
            data_to_save["sleep_time"] = (1 / estimator_instance.exp_freq) - time_tot
            data_to_save["absolute_delay_tcp"] = absolute_delay_tcp
            data_to_save["absolute_time_receive"] = absolute_time_received_dic
            data_to_save["absolute_time_frame"] = absolute_time_frame
            save_results(
                data_to_save,
                estimator_instance.current_time,
                estimator_instance.kin_data_to_track,
                estimator_instance.track_emg,
                estimator_instance.use_torque,
                estimator_instance.result_dir,
                file_name=estimator_instance.result_file_name,
            )
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
        try:
            if target["kin_target"][1][1].shape[2] > estimator_instance.ns_mhe:
                return True
        except:
            if target["kin_target"][1].shape[1] > estimator_instance.ns_mhe:
                return True
        else:
            return False
    else:
        return True


class CustomMhe(MovingHorizonEstimator):
    """
    Class for the custom MHE.
    """

    def __init__(self, **kwargs):
        self.x_ref = None
        self.muscles_ref = None
        self.kin_target = None
        self.slide_size = 1
        self.f_x, self.f_u, = None, None
        self.kalman = None

        super(CustomMhe, self).__init__(**kwargs)

    def advance_window_initial_guess_states(self, sol, **advance_options):
        # x = sol.states["all"][:, :]
        # t = np.arange(0, x.shape[1], 1)
        # f_x = interp1d(t, x, kind='quadratic', fill_value="extrapolate")
        # t_new = np.arange(self.slide_size, x.shape[1] + self.slide_size, 1)
        # new_x = np.copy(f_x(t_new))
        # for i in range(x.shape[0]):
        #     fit = np.polyfit(t, x[i, :], 1)
        #     new_x[i, :] = np.poly1d(fit)(t_new)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(int(self.nlp[0].x_init.init.shape[0])):
        #     plt.subplot(6, 3, i + 1)
        #     plt.plot(self.nlp[0].x_init.init[i, :], label="initial guess")
        #     plt.plot(sol.states["all"][i, :], label="solution")
        #     if i < 9:
        #         plt.plot(self.x_ref[i, :], label="kalman")
        #     # plt.plot(f_x(t_new)[i, :], label="scipy")
        #     # plt.plot(new_x[i], label="numpy")
        # q_est = sol.states["q"][:, :]
        # qdot_est = sol.states["qdot"][:, :]
        # qdot_dif = np.copy(qdot_est)
        # plt.legend()
        #
        # def finite_difference(data):
        #     t = np.linspace(0, data.shape[0] / 100, data.shape[0])
        #     dt = t[1] - t[0]
        #     y = data
        #     dydt = np.gradient(y, dt)
        #     return dydt

        from biosiglive.processing.data_processing import OfflineProcessing
        # for i in range(qdot_est.shape[0]):
        #     qdot_dif[i, :] = finite_difference(q_est[i, :])
            # filtering
            # qdot_dif[i, :] = OfflineProcessing().butter_lowpass_filter(qdot_dif[i, :], 5, 100)
        # plt.figure("q_dot")
        # t_est = np.linspace(0, t[-1], q_est.shape[1])
        # for i in range(9):
        #     plt.subplot(3, 3, i + 1)
        #     plt.plot(qdot_est[i, :], label="qdot_est")
        #     plt.plot(qdot_dif[i, :], label="qdot_dif")
        #     # plt.plot(t, qdot[i, :] * 180 / np.pi, label="qdotkalman")
        #     # plt.title(model.nameDof()[i].to_string())
        # plt.legend()
        # plt.show()

        # self.nlp[0].x_init.init[:, :] = np.concatenate((self.x_ref[:, :], x[9:, :]), axis=0)
        # self.nlp[0].x_init.init[:, :] = np.concatenate(
        #     (sol.states["all"][:, self.slide_size:], sol.states["all"][:, -self.slide_size:]), axis=1)
        # self.slide_size = 1
        self.nlp[0].x_init.init[:, :] = np.concatenate(
            (sol.states["all"][:, self.slide_size:], np.repeat(sol.states["all"][:, -1][:, np.newaxis],
                                                               self.slide_size, axis=1)), axis=1
        )
        # self.nlp[0].x_init.init[:, :] = np.concatenate(
        #     (self.x_ref, sol.states["qdot"]), axis=0
        # )
        # self.nlp[0].x_init.init[:, :] = np.concatenate(
        #     (sol.states["all"][:, self.slide_size:], sol.states["all"][:, :]
        #     ,), axis=1)
        # plot sol.state to see if it is correct in subplots

        return True

    def advance_window_initial_guess_controls(self, sol, **advance_options):
        self.nlp[0].u_init.init[:, :] = np.concatenate(
            (sol.controls["all"][:, self.slide_size:-1], np.repeat(sol.controls["all"][:, -2][:, np.newaxis], self.slide_size, axis=1)), axis=1
        )

        # u = sol.controls["all"][:, :-1]
        # t = np.arange(0, u.shape[1], 1)
        # t_new = np.arange(self.slide_size, u.shape[1] + self.slide_size)
        # new_u = np.copy(u)
        # for i in range(u.shape[0]):
        #     fit = np.polyfit(t, u[i, :], 1)
        #     new_u[i, :] = np.poly1d(fit)(t_new)
        #
        # # self.nlp[0].u_init.init[:, :] = f_u(t_new)
        # self.nlp[0].u_init.init[:, :] = new_u

        # self.nlp[0].u_init.init[:, :] = np.concatenate(
        #     (sol.controls["all"][:, 1:-1], sol.controls["all"][:, -2:-1]
        #     ,), axis=1)
        return True

    def advance_window_bounds_states(self, sol, **advance_options):
        # self.slide_size = 1
        self.nlp[0].u_bounds[:, 0] = sol.controls["all"][:, self.slide_size]
        self.nlp[0].x_bounds.min[:, 0] = sol.states["all"][:, self.slide_size]
        self.nlp[0].x_bounds.max[:, 0] = sol.states["all"][:, self.slide_size]
        return True

    def export_data(self, sol) -> tuple:
        return sol.states["all"][:, self.frame_to_export:self.frame_to_export+self.slide_size], sol.controls["all"][:, self.frame_to_export:self.frame_to_export+self.slide_size]