"""
Stub/wiring-test version of main_qrsmatch_revisit.py for the POP-HT adaptation.

Purpose: confirm the voxelised .alg mesh, limb-only discrepancy function, and
overall QRS-matching loop run end-to-end BEFORE plugging in real electrode
positions or real ECG data. Electrodes and target ECG below are placeholders —
see the TODO markers.

Run this locally first with the small n_tries/n_iterations below.
"""
import numpy as np
import time
import os
import math
import random
import concurrent.futures

import alg_utils
import qrs_matching as qrsm2
import ecg
from constants import *

# ============================================================ CONFIG ===
case_id = "POP-HT_01-08"
DIR = f"/Users/jenny/Google Drive/My Drive/POP-HT/volumetric-meshes/"
MESH_PATH = DIR + f"{case_id}/{case_id}_voxelised.alg"  # anything meshio/pyvista can read (.xdmf, .vtu, .vtk, .msh)
# MESH_PATH = "path/to/your_voxelised_output.alg"   # <-- your .alg from voxelise.py
RUN_DIR = "stub_run_popht"

DX = 2000          # must match the dx you voxelised at (um)
N_TRIES = 8         # small for a fast wiring test; real runs use 128-1024+
N_PROCESSORS = 2
N_ITERATIONS = 20   # small for wiring test; real runs use ~2000
STOP_THRESH = 0.00002
RANDOM_SEED = 0

ITER_DT_S, QRS_SAFETY_S = 0.002, 0.02
MIN_N_ROOT_NODES, MAX_N_ROOT_NODES, ROOT_NODES_DIST_APART_UM = 6, 10, 5000
V_ENDO_MIN, V_ENDO_MAX, V_ENDO_DIFF = 70, 190, 10
V_MYO_MIN, V_MYO_MAX, V_MYO_DIFF = 20, 60, 10
USE_FIBERS, USE_BEST_GUESS = 0, 0
# =========================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(RUN_DIR, exist_ok=True)


# --------------------------------------------------------------------- #
# Limb-only discrepancy function (leads_target may lack V1-V6)
# --------------------------------------------------------------------- #
def calc_discrepancy_limb_leads_only(leads_sim, leads_target):
    limb_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
    leads_present = [l for l in limb_leads if l in leads_target]
    if not leads_present:
        raise ValueError("No limb leads found in leads_target")

    alpha = qrsm2.find_optimal_scaling(
        {l: leads_sim[l] for l in leads_present},
        {l: leads_target[l] for l in leads_present},
    )
    leads_sim_rescaled = {l: leads_sim[l] * alpha for l in leads_present}
    target_qrs_amps = {l: np.max(leads_target[l]) - np.min(leads_target[l]) for l in leads_present}
    leads_discrepancy = {
        l: np.mean(np.abs(leads_sim_rescaled[l] - leads_target[l]) / target_qrs_amps[l])
        for l in leads_present
    }
    return np.mean(list(leads_discrepancy.values()))


# --------------------------------------------------------------------- #
# TODO: replace with real electrode positions from your DICOM/torso work.
# Only LA/RA/LL (indices 0,1,2) matter for limb leads; RL and V1-V6 are
# placeholders (never scored) but must be finite and non-coincident with
# mesh nodes.
# --------------------------------------------------------------------- #
def stub_electrodes_xyz(xs, ys, zs):
    center = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
    span = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()])
    scale = span.max()

    LA = center + np.array([-1.0, 1.0, 0.3]) * scale
    RA = center + np.array([1.0, 1.0, 0.3]) * scale
    LL = center + np.array([0.0, -1.2, -0.5]) * scale
    RL = LL + np.array([50000.0, 0.0, 0.0])   # unused placeholder, offset to avoid coincidence

    # unused placeholders for V1-V6 (never scored by the limb-only discrepancy fn)
    precordial_placeholders = [LA + np.array([i * 20000.0, 0.0, 0.0]) for i in range(1, 7)]

    return [tuple(LA), tuple(RA), tuple(LL), tuple(RL)] + [tuple(p) for p in precordial_placeholders]


# --------------------------------------------------------------------- #
# TODO: replace with your real extracted limb-lead ECG (npy dict of
# {"I": [times_s, signal], ...}). Fabricated placeholder below just
# produces a plausible-shaped QRS complex so the discrepancy/matching
# loop has something non-trivial to run against for the wiring test.
# --------------------------------------------------------------------- #
def stub_leads_targ_in():
    t = np.arange(0, 0.10, ITER_DT_S)
    qrs_shape = np.exp(-((t - 0.04) ** 2) / (2 * 0.008 ** 2))
    leads_targ_in = {}
    limb_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
    rng = np.random.default_rng(RANDOM_SEED)
    for i, lead in enumerate(limb_leads):
        amp = 0.5 + 0.3 * rng.random()
        sign = 1 if i % 2 == 0 else -1
        signal = sign * amp * qrs_shape
        leads_targ_in[lead] = [t, signal]
    return leads_targ_in

# --- Diagnostic plotting helpers ---
def plot_activation_map(xs, ys, zs, activation_times_s):
    import pyvista as pv
    points = np.column_stack([xs, ys, zs]).astype(float)
    cloud = pv.PolyData(points)
    cloud["activation_ms"] = np.asarray(activation_times_s) * 1000
    p = pv.Plotter()
    p.add_mesh(cloud, scalars="activation_ms", cmap="plasma",
               point_size=4, render_points_as_spheres=True)
    p.add_text(f"Activation map (first guess)\n"
               f"range: {cloud['activation_ms'].min():.1f}-{cloud['activation_ms'].max():.1f} ms",
               font_size=10)
    p.show()


def plot_sim_vs_target_ecg(leads_sim, leads_target, times_s, times_target_s):
    import matplotlib.pyplot as plt
    limb_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]

    leads_present = [l for l in limb_leads if l in leads_target]
    alpha = qrsm2.find_optimal_scaling(
        {l: leads_sim[l] for l in leads_present},
        {l: leads_target[l] for l in leads_present},
    )
    print(f"Optimal scaling factor applied to simulated leads: {alpha:.3e}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, lead in zip(axes.flat, limb_leads):
        ax.plot(times_s, leads_sim[lead] * alpha, label="simulated (rescaled)", color="red")
        if lead in leads_target:
            ax.plot(times_target_s, leads_target[lead], label="target", color="black", alpha=0.6)
        ax.set_title(lead)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    runtime_start = time.time()

    discrepancy_metrics = {
        "calc_discrepancy": qrsm2.calc_discrepancy,
        "calc_discrepancy_separate_scaling": qrsm2.calc_discrepancy_separate_scaling,
        "calc_discrepancy_limb_leads_only": calc_discrepancy_limb_leads_only,
    }
    discrep_func = discrepancy_metrics["calc_discrepancy_limb_leads_only"]

    # --- Load voxelised mesh ---
    alg = alg_utils.read_alg_mesh(MESH_PATH)
    xs, ys, zs, *_ = alg_utils.unpack_alg_geometry(alg)
    n_cells = len(xs)
    surface_info_field = alg[6]

    lv_endo_idxs = np.where(surface_info_field == 0)[0]
    rv_endo_idxs = np.where(surface_info_field == 1)[0]
    endo_mask = np.zeros(n_cells)
    endo_mask[lv_endo_idxs], endo_mask[rv_endo_idxs] = 1, 1
    endo_idxs = np.where(endo_mask == 1)[0]
    xs_endo, ys_endo, zs_endo = xs[endo_idxs], ys[endo_idxs], zs[endo_idxs]
    alg_endo = alg_utils.alg_from_xs(xs_endo, ys_endo, zs_endo)

    print(f"Mesh loaded: {n_cells} voxels, {len(endo_idxs)} endocardial "
          f"({len(lv_endo_idxs)} LV, {len(rv_endo_idxs)} RV)")

    # --- Stub electrodes + target ECG ---
    electrodes_xyz = stub_electrodes_xyz(xs, ys, zs)
    leads_targ_in = stub_leads_targ_in()
    leads_target, times_target_s, times_s, total_time_s = qrsm2.prepare_target_leads(
        leads_targ_in, ITER_DT_S, QRS_SAFETY_S
    )

    # --- Precompute geometry for pseudo-ECG ---
    grid_dict = alg_utils.make_grid_dictionary(xs, ys, zs)
    neighbour_arrays, *_ = ecg.get_neighbour_arrays(xs, ys, zs, DX, grid_dict)
    elec_grads = ecg.precompute_elec_grads(xs, ys, zs, electrodes_xyz, DX, neighbour_arrays).astype(np.float32)
    grid_endo_dict = alg_utils.make_grid_dictionary(xs_endo, ys_endo, zs_endo)

    root_nodes_neighbour_dist_um = 2 * ROOT_NODES_DIST_APART_UM
    candidate_root_points, candidate_root_neighbours = qrsm2.mesh_subset_with_dist_constraint(
        alg_endo, ROOT_NODES_DIST_APART_UM, root_nodes_neighbour_dist_um
    )

    candidate_root_node_indices = []
    for point in candidate_root_points:
        candidate_root_node_indices.append(grid_dict[point])

    adjacency_list_26 = ecg.compute_adjacency_displacement(xs, ys, zs, DX, grid_dict, NEIGHBOURS_26)

    v_endos = list(range(V_ENDO_MIN, V_ENDO_MAX + 1, V_ENDO_DIFF))
    v_myos = list(range(V_MYO_MIN, V_MYO_MAX + 1, V_MYO_DIFF))
    v_endos.sort()
    v_myos.sort()

    # --- ADD after candidate_root_node_indices is built, before the main loop ---
    print(f"Candidate root node locations: {len(candidate_root_node_indices)}")
    print(f"Velocity search space: v_endo={v_endos}, v_myo={v_myos} "
          f"({len(v_endos) * len(v_myos)} combinations)")

    param_args = [
        (v_endo, v_myo, adjacency_list_26, endo_mask, USE_FIBERS, candidate_root_node_indices)
        for v_endo in v_endos for v_myo in v_myos
    ]
    batch_size = int(math.ceil(len(param_args) / N_PROCESSORS))
    batched_param_args = list(qrsm2.batcher(param_args, batch_size))

    STOP_AFTER_FIRST_ITER_FOR_DIAGNOSTICS = True  # set False once wiring is confirmed

    print(f"Precomputing {len(param_args)} activation time matrices...")
    all_all_time_matrices = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_PROCESSORS) as executor:
        results = executor.map(qrsm2.compute_time_matrix_batch, batched_param_args)
        for batch_result in results:
            all_all_time_matrices.update(batch_result)

    # --- Initialise root nodes and velocity params ---
    current_iter_params = qrsm2.init_roots_and_vels(
        N_TRIES, MIN_N_ROOT_NODES, MAX_N_ROOT_NODES, candidate_root_node_indices,
        v_endos, v_myos, params_best_guess=None, use_best_guess=False
    )

    mutated_params, all_ids_and_diff_scores = {}, {}
    iter_median_scores = []

    for iter_no in range(N_ITERATIONS):
        n_tries = len(current_iter_params)
        n_per_batch = max(1, int(round(n_tries / N_PROCESSORS)))
        tries = np.arange(n_tries)
        print(f"=== Iter {iter_no}: {n_tries} tries ===", flush=True)

        all_electrodes, activation_times_s = qrsm2.batch_qrs_runner(
            n_tries, n_per_batch, qrsm2.pseudo_ecg_qrs, times_s, qrsm2.ap_heaviside,
            electrodes_xyz, elec_grads, DX, total_time_s, neighbour_arrays,
            all_all_time_matrices, current_iter_params
        )

        # --- ADD: diagnostics on the very first try of the very first iteration ---
        if iter_no == 0:
            first_try_key = list(current_iter_params.keys())[0]
            first_root_params = current_iter_params[first_try_key]
            print(f"\nFirst guess params: v_endo/v_myo={first_root_params[0]}, "
                  f"n_root_nodes={len(first_root_params[1])}, "
                  f"root_indices={first_root_params[1]}")

            print(f"Activation time range for first guess: "
                  f"{min(activation_times_s[first_try_key]) * 1000:.1f} - "
                  f"{max(activation_times_s[first_try_key]) * 1000:.1f} ms")

            plot_activation_map(xs, ys, zs, activation_times_s[first_try_key])
            raw = all_electrodes[first_try_key]
            print(f"Raw electrode signal shape: {raw.shape}")
            for i, name in enumerate(["LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6"]):
                print(f"  {name}: min={raw[i].min():.3e}, max={raw[i].max():.3e}, "
                      f"any_nonzero={np.any(raw[i] != 0)}")

            leads_sim_first = ecg.ten_electrodes_to_twelve_leads(all_electrodes[first_try_key])
            plot_sim_vs_target_ecg(leads_sim_first, leads_target, times_s, times_target_s)

            if STOP_AFTER_FIRST_ITER_FOR_DIAGNOSTICS:
                print("Stopping after first-guess diagnostics (STOP_AFTER_FIRST_ITER_FOR_DIAGNOSTICS=True).")
                return

        population_ids, population_ids_check = {}, {}
        for i_try in tries:
            param_id = qrsm2.hash_qrs_param(current_iter_params[i_try])
            population_ids[i_try] = param_id
            population_ids_check[param_id] = 1

        all_leads_sim, population_diff_scores = qrsm2.analyse_sim_qrs_leads(
            all_electrodes, leads_target, times_s, times_target_s, discrep_func
        )

        for i_try, params in current_iter_params.items():
            param_id = qrsm2.hash_qrs_param((params[0], tuple(sorted(params[1]))))
            all_ids_and_diff_scores[param_id] = [population_diff_scores[i_try], iter_no]

        scores = list(population_diff_scores.values())
        iter_median_scores.append(np.median(scores))

        keys_below, keys_above = ecg.tie_aware_proportional_split(population_diff_scores, 87.5)
        mutated_params = qrsm2.mutate_pop_params(
            keys_above, keys_below, current_iter_params, alg, grid_dict,
            candidate_root_node_indices, candidate_root_neighbours, v_endos, v_myos,
            all_ids_and_diff_scores
        )

        next_iter_params, next_iter_tries_ct = {}, 0
        for key, params in mutated_params.items():
            param_id = qrsm2.hash_qrs_param((params[0], tuple(sorted(params[1]))))
            if param_id not in all_ids_and_diff_scores:
                next_iter_params[next_iter_tries_ct] = params
                next_iter_tries_ct += 1
        current_iter_params = next_iter_params if next_iter_params else current_iter_params

        min_diff_score = min(population_diff_scores.values())
        print(f"Min score: {round(min_diff_score, 5)}, unique params tested: {len(all_ids_and_diff_scores)}")

        converged, _ = ecg.runtime_stop_condn(iter_no, iter_median_scores, window_size=5, stop_thresh=STOP_THRESH)
        if converged:
            print("Converged, stopping.")
            break

    print(f"Done in {time.time() - runtime_start:.1f}s")


if __name__ == "__main__":
    main()