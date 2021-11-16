import numpy as np
from math import *
import csv
import warnings
from casadi import MX, Function, horzcat
import biorbd_casadi as biorbd
from bioptim import (
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    InterpolationType,
    Bounds,
    Node,
    MovingHorizonEstimator,
    Solver,
)


def states_to_markers(biorbd_model, states):
    nq = biorbd_model.nbQ()
    n_mark = biorbd_model.nbMarkers()
    q = MX.sym("q", nq)
    markers_func = biorbd.to_casadi_func("makers", biorbd_model.markers, q)
    return np.array(markers_func(states[:nq, :])).reshape((3, n_mark, -1), order="F")


# Use biorbd function for inverse kinematics
def markers_fun(biorbd_model):
    qMX = MX.sym("qMX", biorbd_model.nbQ())
    return Function(
        "markers", [qMX], [horzcat(*[biorbd_model.markers(qMX)[i].to_mx() for i in range(biorbd_model.nbMarkers())])]
    )


# Return muscle force
def muscles_forces(q, qdot, act, controls, model, nlp, use_excitation=False):
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
        [muscles_forces(qMX, dqMX, aMX, uMX, biorbd_model, nlp=nlp, use_excitation=use_excitation)],
        ["qMX", "dqMX", "aMX", "uMX"],
        ["Force"],
    ).expand()


# Return mean RMSE
def compute_err(
    init_offset,
    final_offset,
    Ns_mhe,
    X_est,
    U_est,
    Ns,
    model,
    nlp,
    q,
    dq,
    tau,
    activations,
    excitations,
    nbGT,
    ratio=1,
    use_excitation=False,
    full_windows=False,
):
    # All variables
    model = model
    get_force = force_func(model, nlp, use_excitation=use_excitation)
    get_markers = markers_fun(model)
    err = dict()
    offset = final_offset - Ns_mhe
    q_ref = q[:, 0 : Ns + 1 : ratio]
    dq_ref = dq[:, 0 : Ns + 1 : ratio]
    tau_ref = tau[:, 0:Ns:ratio]
    if use_excitation is not True:
        muscles_ref = activations[:, 0:Ns:ratio]
    else:
        muscles_ref = excitations[nbGT:, 0:Ns:ratio]
    sol_mark = np.zeros((3, model.nbMarkers(), ceil((Ns + 1) / ratio) - Ns_mhe))
    sol_mark_ref = np.zeros((3, model.nbMarkers(), ceil((Ns + 1) / ratio) - Ns_mhe))

    # Compute RMSE
    err["q"] = np.sqrt(
        np.square(X_est[: model.nbQ(), init_offset:-offset] - q_ref[:, init_offset:-final_offset]).mean(axis=1)
    ).mean()
    err["q_dot"] = np.sqrt(
        np.square(
            X_est[model.nbQ() : model.nbQ() * 2, init_offset:-offset] - dq_ref[:, init_offset:-final_offset]
        ).mean(axis=1)
    ).mean()
    err["tau"] = np.sqrt(
        np.square(U_est[:nbGT, init_offset:-offset] - tau_ref[:nbGT, init_offset:-final_offset]).mean(axis=1)
    ).mean()
    err["muscles"] = np.sqrt(
        np.square(U_est[nbGT:, init_offset:-offset] - muscles_ref[:, init_offset:-final_offset]).mean(axis=1)
    ).mean()

    # Get marker and compute RMSE
    len_x = int(ceil((Ns + 1) / ratio) - Ns_mhe) if full_windows is not True else Ns + 1
    len_u = int(ceil(Ns / ratio) - Ns_mhe) if full_windows is not True else Ns
    for i in range(len_x):
        sol_mark[:, :, i] = get_markers(X_est[: model.nbQ(), i])
    sol_mark_tmp = np.zeros((3, sol_mark_ref.shape[1], Ns + 1))
    for i in range(Ns + 1):
        sol_mark_tmp[:, :, i] = get_markers(q[:, i])
    sol_mark_ref = sol_mark_tmp[:, :, 0 : Ns + 1 : ratio]
    err["markers"] = np.sqrt(
        np.square(sol_mark[:, :, init_offset:-offset] - sol_mark_ref[:, :, init_offset:-final_offset])
        .sum(axis=0)
        .mean(axis=1)
    ).mean()

    # Get muscle force and compute RMSE
    force_ref_tmp = np.ndarray((model.nbMuscles(), Ns))
    force_est = np.ndarray((model.nbMuscles(), len_u))
    if use_excitation is not True:
        a_est = np.zeros((model.nbMuscles(), Ns))
    else:
        a_est = X_est[-model.nbMuscles() :, :]

    for i in range(model.nbMuscles()):
        for j in range(len_u):
            force_est[i, j] = get_force(
                X_est[: model.nbQ(), j], X_est[model.nbQ() : model.nbQ() * 2, j], a_est[:, j], U_est[nbGT:, j]
            )[i, :]
    get_force = force_func(model, nlp, use_excitation=False)
    for i in range(model.nbMuscles()):
        for k in range(Ns):
            force_ref_tmp[i, k] = get_force(q[:, k], dq[:, k], activations[:, k], excitations[nbGT:, k])[i, :]
    force_ref = force_ref_tmp[:, 0:Ns:ratio]
    err["force"] = np.sqrt(
        np.square(force_est[:, init_offset:-offset] - force_ref[:, init_offset:-final_offset]).mean(axis=1)
    ).mean()

    return err


# Return which iteration has not converged
def convert_txt_output_to_list(file, nbco, nbmark, nbemg, nbtries):
    conv_list = [[[[[] for i in range(nbtries)] for j in range(nbemg)] for k in range(nbmark)] for l in range(nbco)]
    with open(file) as f:
        fdel = csv.reader(f, delimiter=";", lineterminator="\n")
        for line in fdel:
            if line[0] == "7":
                try:
                    conv_list[int(line[1])][int(line[2])][int(line[3])][int(line[4])].append(line[5])
                except:
                    warnings.warn(f"line {line} ignored")
    return conv_list


# New function for bioptim.mhe
def define_objective(
    use_excitation, use_torque, TRACK_EMG, muscles_target, kin_target, biorbd_model, kin_data_to_track="markers"
):
    # muscle_track_idx = range(0, 34)
    muscle_track_idx = [0, 1, 2, 3, 4, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26]
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
    if TRACK_EMG:
        weight = {
            "track_markers": 1000000,
            "track_q": 100000,
            "min_control": 100,
            "min_dq": 10,
            "min_q": 1,
            "min_torque": 10,
            "min_act": 100,
            "track_EMG": 1000,
        }
        # muscles_target =
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_CONTROL,
            weight=weight["track_EMG"],
            target=muscle_target_reduce,
            index=muscle_track_idx,
            key="muscles",
            multi_thread=False,
        )

        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weight["min_control"], index=muscle_min_idx, key="muscles", multi_thread=False
        )
    else:
        weight = {"track_markers": 10000000,
            "track_q": 10000, "min_dq": 10, "min_q": 1, "min_torque": 10, "min_act": 10}
        if use_excitation is not True:
            weight["min_control"] = 100
        else:
            weight["min_control"] = 1000
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weight["min_control"], key="muscles", multi_thread=False
        )

    if use_torque:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=weight["min_torque"], key="tau", multi_thread=False
        )

    kin_funct = ObjectiveFcn.Lagrange.TRACK_STATE if kin_data_to_track == "q" else ObjectiveFcn.Lagrange.TRACK_MARKERS
    kin_weight = weight["track_markers"] if kin_data_to_track == "markers" else weight["track_q"]
    if kin_data_to_track == 'markers':
        objectives.add(kin_funct, weight=kin_weight, target=kin_target, node=Node.ALL, multi_thread=False)
    elif kin_data_to_track == 'q':
        objectives.add(kin_funct, weight=kin_weight, target=kin_target, key='q', node=Node.ALL, multi_thread=False)

    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weight["min_q"],
        index=np.array(range(biorbd_model.nbQ())),
        key="q",
        node=Node.ALL,
        multi_thread=False,
    )
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weight["min_dq"],
        index=np.array(range(biorbd_model.nbQ())),
        key="qdot",
        node=Node.ALL,
        multi_thread=False,
    )
    if use_excitation is True:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            weight=weight["min_act"],
            index=np.array(range(biorbd_model.nbQ() * 2, biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles())),
            key="muscles",
        )
    return objectives


def prepare_mhe(
    biorbd_model, objectives, window_len, window_duration, x0, use_torque=False, use_excitation=False, nb_threads=4
):
    # Model path
    biorbd_model = biorbd_model
    nbGT = biorbd_model.nbGeneralizedTorque() if use_torque else 0
    nbGT = nbGT
    nbMT = biorbd_model.nbMuscleTotal()
    tau_min, tau_max, tau_init = -10, 10, 0
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
    x_init = InitialGuess(np.tile(x0, (window_len + 1, 1)).T, interpolation=InterpolationType.EACH_FRAME)

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

    solver = Solver.ACADOS()
    solver.set_convergence_tolerance(1e-5)
    solver.set_integrator_type("IRK")
    solver.set_nlp_solver_type("SQP")
    solver.set_sim_method_num_steps(1)
    solver.set_print_level(1)
    solver.set_maximum_iterations(20)

    # ------------- #
    return mhe, solver


class CustomMhe(MovingHorizonEstimator):
    def __init__(self, **kwargs):

        self.init_w_kalman = False
        self.x_ref = []
        self.muscles_target = []
        self.kin_target = []

        super(CustomMhe, self).__init__(**kwargs)

    # def advance_window(self, sol, steps: int = 0):
    #     if self.nlp[0].x_bounds.type != InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
    #         if self.nlp[0].x_bounds.type == InterpolationType.CONSTANT:
    #             x_min = np.repeat(self.nlp[0].x_bounds.min[:, :1], 3, axis=1)
    #             x_max = np.repeat(self.nlp[0].x_bounds.max[:, :1], 3, axis=1)
    #             self.nlp[0].x_bounds = Bounds(x_min, x_max)
    #         else:
    #             raise NotImplementedError(
    #                 "The MHE is not implemented yet for x_bounds not being "
    #                 "CONSTANT or CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT"
    #             )
    #         self.nlp[0].x_bounds.check_and_adjust_dimensions(self.nlp[0].states.shape, 3)
    #     self.nlp[0].x_bounds[:, 0] = sol.states["all"][:, 1]
    #     x = sol.states["all"]
    #     u = sol.controls["all"][:, :-1]
    #     if self.init_w_kalman:
    #         x0 = np.hstack((x[:, 1:], self.x_ref[:, -1:]))
    #     else:
    #         x0 = np.hstack((x[:, 1:], x[:, -1:]))  # discard oldest estimate of the window, duplicates youngest
    #     u0 = np.hstack((u[:, 1:], u[:, -1:]))
    #     x_init = InitialGuess(x0, interpolation=InterpolationType.EACH_FRAME)
    #     u_init = InitialGuess(u0, interpolation=InterpolationType.EACH_FRAME)
    #     self.update_initial_guess(x_init, u_init)
