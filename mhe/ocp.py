from .utils import *
from time import time, sleep
import biorbd_casadi as biorbd
from scipy import interpolate
from biosiglive.data_processing import read_data
import numpy as np
import datetime
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
    ParameterList,

)


def muscle_forces(q, qdot, act, controls, model, use_excitation=False):
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        if use_excitation is not True:
            muscles_states[k].setActivation(controls[k])
        else:
            muscles_states[k].setExcitation(controls[k])
            muscles_states[k].setActivation(act[k])
    muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx()
    return muscles_force


# Return biorbd muscles force function
def force_func(biorbd_model, use_excitation=False):
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


def get_reference_data(file_path):
    if file_path[-4:] == ".mat":
        mat = sio.loadmat(file_path)
        x_ref, markers, muscles = mat['kalman'], mat["markers"], mat["emg_proc"]

    else:
        mat = read_data(file_path)
        try:
            x_ref, markers, muscles = mat['kalman'], mat["kin_target"], mat["muscles_target"]
        except:
            x_ref, markers, muscles = mat['kalman'], mat["markers"], mat["emg"]

    return x_ref, markers, muscles


def define_objective(
    weights, use_excitation, use_torque, track_emg, muscles_target, kin_target, biorbd_model, kin_data_to_track="markers",
        muscle_track_idx=()
):
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
        kin_target_thorax = kin_target[:3, :4, :]
        kin_target_bodies = kin_target[:3, 4:, :]
        idx_thorax = [0, 1, 2, 3]
        thorax_weight_ratio = 1000
        idx_bodies = list(range(4, biorbd_model.nbMarkers()))
        objectives.add(kin_funct,
                       weight=kin_weight / thorax_weight_ratio,
                       target=kin_target_thorax,
                       marker_index=idx_thorax,
                       node=Node.ALL,
                       multi_thread=False)
        objectives.add(kin_funct,
                       weight=kin_weight,
                       target=kin_target_bodies,
                       marker_index=idx_bodies,
                       node=Node.ALL,
                       multi_thread=False)

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


def modify_isometric_force(biorbd_model, value, fiso_init):
    for k in range(biorbd_model.nbMuscles()):
        biorbd_model.muscle(k).characteristics().setForceIsoMax(value[k] * fiso_init[k])


def my_target_function(ocp: OptimalControlProgram, value: MX) -> MX:
    """
    The target function is a penalty function.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference the user can use to access all the elements of the ocp
    value: MX
        The parameter variable
    Returns
    -------
    The value to minimize. If a target value exist (target parameters) it is automatically added, and therefore
    should not be added by hand here (that is, the next line should not read: return value - target)
    """

    return value

def prepare_problem(
    model_path, objectives, window_len, window_duration, x0, use_torque=False, use_excitation=False, nb_threads=8,
        is_mhe=True,
        use_parameters=False,
        solver_options={}
):
    # Model path

    biorbd_model = biorbd.Model(model_path)
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

    x_bounds[0].min[biorbd_model.nbQ():, :] = [[-20] * 3] * biorbd_model.nbQ()
    x_bounds[0].max[biorbd_model.nbQ():, :] = [[20] * 3] * biorbd_model.nbQ()

    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * nbGT + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [tau_max] * nbGT + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    # Initial guesses
    if is_mhe:
        x_init = InitialGuess(np.concatenate((x0[:, :window_len+1], np.zeros((x0.shape[0], window_len+1)))),
                              interpolation=InterpolationType.EACH_FRAME)

    else:
        x_init = InitialGuess(np.concatenate((x0[:, :window_len+1], np.zeros((x0.shape[0], window_len+1)))),
                              interpolation=InterpolationType.EACH_FRAME)

    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)
    u_init = InitialGuess(np.tile(u0, (window_len, 1)).T, interpolation=InterpolationType.EACH_FRAME)

    if use_parameters:
        nb_mus = biorbd_model.nbMuscles()
        # Get initial isometric forces
        f_iso = []
        for k in range(nb_mus):
            f_iso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())

        # Define the parameter to optimize
        bound_p_iso = Bounds(
            [0.5] * nb_mus, [2.5] * nb_mus, interpolation=InterpolationType.CONSTANT)
        p_init = [1] * nb_mus
        from bioptim import Objective
        parameter_objective_functions = Objective(
            my_target_function, weight=10000, quadratic=True, custom_type=ObjectiveFcn.Parameter, target=[1] * nb_mus
        )
        p_iso_init = InitialGuess(p_init)

        parameters = ParameterList()
        parameters.add(
            "p_iso",  # The name of the parameter
            modify_isometric_force,  # The function that modifies the biorbd model
            p_iso_init,
            bound_p_iso,  # The bounds
            size=nb_mus,  # The number of elements this particular parameter vector has
            fiso_init=f_iso,
            penalty_list=parameter_objective_functions,
        )
    else:
        parameters = None
    biorbd_model = biorbd.Model(model_path)

    if is_mhe:
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
            parameters=parameters
        )
    else:
        problem = OptimalControlProgram(
            biorbd_model=biorbd_model,
            dynamics=dynamics,
            objective_functions=objectives,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            n_threads=nb_threads,
            n_shooting=window_len,
            phase_time=window_duration,
            parameters=parameters,
            use_sx=True,
        )

    solver = Solver.ACADOS()
    solver.set_convergence_tolerance([1e-4, 1e-4, 1e-4, 1e-1])
    solver.set_integrator_type("IRK")
    solver.set_qp_solver("PARTIAL_CONDENSING_HPIPM")

    if is_mhe:
        solver.set_nlp_solver_type("SQP_RTI")
        solver.set_print_level(0)
        solver.set_maximum_iterations(50)
        for key in solver_options.keys():
            solver.set_option(val=solver_options[key], name=key)


    else:
        solver.set_nlp_solver_type("SQP_RTI")
        solver.set_print_level(0)
        solver.set_maximum_iterations(300)
    solver.set_sim_method_num_steps(1)

    # ------------- #
    return problem, solver


def configure_weights(track_emg=True, is_mhe=True, kin_data='markers', use_excitation=False):
    # Working for abd and flex
    weights = {
        "track_markers": 10000000,
        "track_q": 100000,
        "min_control": 1000,
        "min_dq": 100,
        "min_q": 10,
        "min_torque": 1000,
        "min_act": 1,
        "track_emg": 100000
    }
    # weights = {
    #     "track_markers": 10000000,
    #     "track_q": 100000,
    #     "min_control": 100,
    #     "min_dq": 100,
    #     "min_q": 10,
    #     "min_torque": 100,
    #     "min_act": 1,
    #     "track_emg": 1000
    # }

    # weights = {
    #     "track_markers": 100000,
    #     "min_control": 1000,
    #     "min_dq": 1000,
    #     "min_q": 10,
    #     "min_torque": 10,
    #     "min_act": 0.1,
    #     "track_emg": 100
    # }

    # if is_mhe:
    #     if not track_emg:
    #         weights["track_markers"] = 100000
    #         weights["min_dq"] = 100
    #         weights["min_q"] = 10
    #         weights["min_control"] = 1000
    #
    #     if kin_data == "q":
    #         if track_emg:
    #             weights["track_q"] = 1000000
    #             weights["min_q"] = 10
    #             weights["min_torque"] = 100
    #         else:
    #             weights["track_q"] = 100000
    #             weights["min_dq"] = 10
    #             weights["min_q"] = 1
    #             weights["min_torque"] = 10
    # else:
    #     if track_emg:
    #         weights["track_markers"] = 10000000
    #         weights["min_dq"] = 10
    #         weights["min_q"] = 1
    #         weights["min_control"] = 1000
    #         weights["track_emg"] = 10000
    #         weights["min_torque"] = 10
    #     else:
    #         weights["track_markers"] = 1000
    #         weights["min_dq"] = 10
    #         weights["min_q"] = 1
    #         weights["min_control"] = 100
    #
    #     if kin_data == "q":
    #         if track_emg:
    #             weights["track_q"] = 10000
    #             weights["min_dq"] = 100
    #             weights["min_q"] = 10
    #             weights["min_torque"] = 10
    #             weights["min_control"] = 100
    #             weights["track_emg"] = 1000
    #         else:
    #             weights["track_q"] = 1000000
    #             weights["min_dq"] = 10
    #             weights["min_q"] = 1
    #             weights["min_torque"] = 10
    return weights

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
            markers_target_idx.append(i)
        elif mhe.nlp[0].J[i].name == "MINIMIZE_STATE" and mhe.nlp[0].J[i].target is not None:
            q_target_idx = i
    kin_target_idx = q_target_idx if kin_data_to_track == "q" else markers_target_idx

    # Define target
    muscle_target = muscles_ref[:, slide_size*t: (ns_mhe + 1 + slide_size*t)] if offline else muscles_ref

    # Markers target:
    markers_target_thorax = markers_ref[:3, :4, slide_size*t: (ns_mhe + 1 + slide_size*t)] if offline else markers_ref[:3, :4, :]
    markers_target_bodies = markers_ref[:3, 4:, slide_size*t: (ns_mhe + 1 + slide_size*t)] if offline else markers_ref[:3, 4:, :]

    # Angle target:
    q_target = x_ref[:, slide_size*t: (ns_mhe + 1 + slide_size*t)] if offline else x_ref
    q_target = np.concatenate((q_target, np.zeros((nbQ, q_target.shape[1]))), axis=0) if x_ref.shape[0] < nbQ else q_target

    if kin_data_to_track == "q":
        kin_target = q_target
    else:
        kin_target_thorax = markers_target_thorax
        kin_target_bodies = markers_target_bodies
        kin_target = [kin_target_thorax, kin_target_bodies]

    target = {"kin_target": [kin_target_idx, kin_target]}
    if track_emg:
        target["muscle_target"] = [muscles_target_idx, muscle_target]
    return target


def update_mhe(mhe, t, sol, estimator_instance, muscle_track_idx, initial_time, offline_data=None):
    tic = time()
    if sol:
        print(sol.status)
    # if t == 1:
    #     import sys
    #     sys.stdin = os.fdopen(0)
    #     input("Ready to run estimator ? (press any key to continue)")
    absolute_time_frame = 0
    absolute_delay_tcp = 0
    absolute_time_received_dic = 0
    if estimator_instance.test_offline:
        x_ref, markers_target, muscles_target = offline_data
        t_to_stop = int(x_ref.shape[1] * estimator_instance.interpol_factor / estimator_instance.slide_size)

    else:
        if estimator_instance.data_process:
            while True:
                try:
                    data = estimator_instance.data_queue.get_nowait()
                    x_ref = np.array(data["kalman"])
                    markers_target = np.array(data["markers"])
                    muscles_target = np.array(data["emg"])
                    break
                except:
                    pass
        else:
            data = estimator_instance.get_data(t)
            x_ref = np.array(data["kalman"])
            markers_target = np.array(data["markers"])
            muscles_target = np.array(data["emg"])
            absolute_time_frame = data["absolute_time_frame"]
            absolute_time_received = datetime.datetime.now()
            absolute_time_received_dic = {"day": absolute_time_received.day,
                                          "hour": absolute_time_received.hour,
                                          "minute": absolute_time_received.minute,
                                          "second": absolute_time_received.second,
                                          "millisecond": int(absolute_time_received.microsecond/1000),
                                       }
            absolute_delay_tcp = {}
            for key in absolute_time_frame.keys():
                absolute_delay_tcp[key] = absolute_time_received_dic[key] - absolute_time_frame[key]

    # interpolate target
    if estimator_instance.interpol_factor != 1 and estimator_instance.is_mhe:
        # x_ref
        x = np.linspace(0, x_ref.shape[1] / 100, x_ref.shape[1])
        f_x = interpolate.interp1d(x, x_ref)
        x_new = np.linspace(0, x_ref.shape[1] / 100, int(x_ref.shape[1] * estimator_instance.interpol_factor))
        x_ref = f_x(x_new)

        # markers_ref
        markers_ref = np.zeros(
            (3, markers_target.shape[1], int(markers_target.shape[2] * estimator_instance.interpol_factor)))
        for i in range(3):
            x = np.linspace(0, markers_target.shape[2] / 100, markers_target.shape[2])
            f_mark = interpolate.interp1d(x, markers_target[i, :, :])
            x_new = np.linspace(0, markers_target.shape[2] / 100,
                                int(markers_target.shape[2] * estimator_instance.interpol_factor))
            markers_ref[i, :, :] = f_mark(x_new)

        # muscle_target
        x = np.linspace(0, muscles_target.shape[1] / 100, muscles_target.shape[1])
        f_mus = interpolate.interp1d(x, muscles_target)
        x_new = np.linspace(0, muscles_target.shape[1] / 100,
                            int(muscles_target.shape[1] * estimator_instance.interpol_factor))
        muscles_target = f_mus(x_new)
    else:
        markers_ref = markers_target
    # muscles_ref = muscles_target
    muscles_ref = np.zeros((len(estimator_instance.muscle_track_idx), int(muscles_target.shape[1])))
    muscles_ref[[0, 1, 2], :] = muscles_target[0, :]
    muscles_ref[[3], :] = muscles_target[1, :]
    muscles_ref[4, :] = muscles_target[2, :]
    muscles_ref[5, :] = muscles_target[3, :]
    muscles_ref[[6, 7], :] = muscles_target[4, :]
    muscles_ref[[8, 9, 10], :] = muscles_target[5, :]
    muscles_ref[[11], :] = muscles_target[6, :]
    muscles_ref[[12], :] = muscles_target[7, :]
    muscles_ref[[13], :] = muscles_target[8, :]
    muscles_ref[[14], :] = muscles_target[9, :]
    muscles_ref = muscles_ref / np.repeat(estimator_instance.mvc_list, muscles_target.shape[1]).reshape(
        len(estimator_instance.mvc_list), muscles_target.shape[1])
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
        mhe.x_ref = x_ref[:, estimator_instance.slide_size*t:(estimator_instance.ns_mhe + 1 + estimator_instance.slide_size*t)]
        mhe.muscles_ref = muscles_ref[:, estimator_instance.slide_size*t: estimator_instance.ns_mhe + estimator_instance.slide_size*t]
    else:
        mhe.x_ref = x_ref
        mhe.muscles_ref = muscles_ref

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
            sol, estimator_instance.get_force, estimator_instance.nbMT, estimator_instance.use_excitation
        )

        if estimator_instance.data_to_show:
            dic_to_put = {"t": t, "force_est": force_est.tolist(), "q_est": q_est.tolist()}
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
                "solver_options": estimator_instance.solver_options
            }
            if t == 1:
                data_to_save["Nmhe"] = estimator_instance.ns_mhe
            data_to_save["muscles_target"] = target["muscle_target"][1][:, -1:] if "muscle_target" in target.keys() else 0
            if estimator_instance.kin_data_to_track == 'q':
                kin_target = target["kin_target"][1][:, -2:-1]
            else:
                kin_target = []
                if isinstance(target["kin_target"][1], list):
                    kin_target = np.concatenate((target["kin_target"][1][0][:, :, -2:-1],
                                                 target["kin_target"][1][1][:, :, -2:-1]), axis=1)
                else:
                    kin_target.append(target["kin_target"][1][:, :, -2:-1])

            data_to_save["kin_target"] = kin_target
            data_to_save["sol_freq"] = 1 / time_tot
            data_to_save["exp_freq"] = estimator_instance.exp_freq
            data_to_save["sleep_time"] = (1 / estimator_instance.exp_freq) - time_tot
            data_to_save["absolute_delay_tcp"] = absolute_delay_tcp
            data_to_save["absolute_time_receive"] = absolute_time_received_dic
            data_to_save["absolute_time_frame"] = absolute_time_frame
            save_results(sol,
                         data_to_save,
                         estimator_instance.current_time,
                         estimator_instance.kin_data_to_track,
                         estimator_instance.track_emg,
                         estimator_instance.use_torque,
                         estimator_instance.use_excitation,
                         estimator_instance.result_dir)

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
        # if t == t_to_stop - 1:
        if t == 400:
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

        super(CustomMhe, self).__init__(**kwargs)

    def advance_window(self, sol, steps: int = 0, **advance_options):
        x = sol.states["all"]
        u = sol.controls["all"][:, :-1]
        if self.init_w_kalman:
            x0 = np.hstack((x[:, self.slide_size:], self.x_ref[:, -self.slide_size:]))
        else:
            x0 = np.hstack((x[:, self.slide_size:], x[:, -self.slide_size:]))  # discard oldest estimate of the window, duplicates youngest
        u0 = np.hstack((u[:, self.slide_size:], u[:, -self.slide_size:]))
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
    x_bounds[0].min[biorbd_model.nbQ():, :] = [-30] * biorbd_model.nbQ()
    x_bounds[0].max[biorbd_model.nbQ():, :] = [30] * biorbd_model.nbQ()

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
