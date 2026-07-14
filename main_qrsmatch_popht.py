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
ecg_case_id = "id_008"
DIR = f"/Users/jenny/Google Drive/My Drive/POP-HT/"
MESH_PATH = DIR + f"volumetric-meshes/{case_id}/{case_id}_voxelised.alg"  # anything meshio/pyvista can read (.xdmf, .vtu, .vtk, .msh)
# MESH_PATH = "path/to/your_voxelised_output.alg"   # <-- your .alg from voxelise.py
RUN_DIR = "run_popht"
ELECTRODE_PATH = DIR + f"ecgs/{case_id}_electrodes_xyz.npy"
REP_ECG_BEAT_DIR = DIR + f"ecgs/ecg_representative_beat_output/{ecg_case_id}"
SANITY_CHECK_DIR = os.path.join(RUN_DIR, "sanity_checks", case_id)
os.makedirs(SANITY_CHECK_DIR, exist_ok=True)

DX = 2000          # must match the dx you voxelised at (um)
# N_TRIES = 128
N_TRIES = 256 #1024 - on ARC
N_ITERATIONS = 200
# N_TRIES = 8         # small for a fast wiring test; real runs use 128-1024+
N_PROCESSORS = 2
# N_ITERATIONS = 20   # small for wiring test; real runs use ~2000
STOP_THRESH = 0.00002
RANDOM_SEED = 0

SAVE_SIM_ECG_EVERY_N_ITERS = 10
CONVERGENCE_HISTORY_DIR = os.path.join(SANITY_CHECK_DIR, "convergence_history")
os.makedirs(CONVERGENCE_HISTORY_DIR, exist_ok=True)
RUN_FIRST_ITER_DIAGNOSTICS = False

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

def load_electrodes_xyz(electrodes_path):
    """Load the real rule-based electrode positions saved by
    rule_based_electrode_placement.py. Expects a list/array of 10
    (x, y, z) tuples in [LA, RA, LL, RL, V1..V6] order, already in um."""
    electrodes_xyz = np.load(electrodes_path, allow_pickle=True)
    electrodes_xyz = [tuple(e) for e in electrodes_xyz]
    if len(electrodes_xyz) != 10:
        raise ValueError(f"Expected 10 electrode positions, got {len(electrodes_xyz)}")
    return electrodes_xyz

def load_leads_targ_in(rep_beat_dir, limb_leads=("I", "II", "III", "aVR", "aVL", "aVF"),
                        sampling_rate=500.0, reference_lead="II", isolate_qrs=True, run_qc_plot=True):
    import pandas as pd
    leads_targ_in_raw = {}
    for lead in limb_leads:
        csv_path = os.path.join(rep_beat_dir, f"{lead}_representative_beat.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping {lead}")
            continue
        df = pd.read_csv(csv_path)
        times_s = df["time_rel_to_r_s"].to_numpy()
        signal = df["amplitude_mV_representative"].to_numpy()
        leads_targ_in_raw[lead] = [times_s, signal]

    if not leads_targ_in_raw:
        raise ValueError(f"No representative-beat CSVs found in {rep_beat_dir}")

    if not isolate_qrs:
        return leads_targ_in_raw

    leads_targ_in_trimmed, window_info = isolate_qrs_window_shared(
        leads_targ_in_raw, reference_lead, sampling_rate
    )

    if run_qc_plot:
        plot_qrs_isolation_qc(leads_targ_in_raw, window_info, sampling_rate)

    return leads_targ_in_trimmed

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
    p.show(screenshot=os.path.join(SANITY_CHECK_DIR, "activation_map_first_guess.png"))


def plot_sim_vs_target_ecg(leads_sim, leads_target, times_s, times_target_s):
    import matplotlib.pyplot as plt
    limb_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]

    leads_present = [l for l in limb_leads if l in leads_target]

    # Use the UNCLIPPED scaling factor for visualization -- find_optimal_scaling
    # clips negative alpha to 0 (correct for the actual discrepancy metric,
    # since a negative scale isn't a meaningful "fit"), but that makes a
    # genuinely anticorrelated first guess invisible on a plot. For diagnostic
    # purposes we want to actually SEE the mismatch, including sign flips.
    sim_concat = np.concatenate([leads_sim[l] for l in leads_present])
    target_concat = np.concatenate([leads_target[l] for l in leads_present])
    denom = np.sum(sim_concat ** 2)
    alpha_unclipped = np.sum(sim_concat * target_concat) / denom if denom != 0 else 0.0

    alpha_clipped = qrsm2.find_optimal_scaling(
        {l: leads_sim[l] for l in leads_present},
        {l: leads_target[l] for l in leads_present},
    )

    print(f"Unclipped scaling factor (used for plotting): {alpha_unclipped:.3e}")
    print(f"Clipped scaling factor (used for actual scoring): {alpha_clipped:.3e}")
    if alpha_unclipped < 0:
        print("  -> NEGATIVE: this first guess is anticorrelated with target "
              "(expected for a random initial guess; the search should move past this)")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, lead in zip(axes.flat, limb_leads):
        ax.plot(times_s, leads_sim[lead] * alpha_unclipped, label="simulated (unclipped scale)", color="red")
        if lead in leads_target:
            ax.plot(times_target_s, leads_target[lead], label="target", color="black", alpha=0.6)
        ax.set_title(lead)
        ax.legend(fontsize=8)
    fig.suptitle(f"unclipped alpha={alpha_unclipped:.2e} (for visualization only -- "
                 f"actual discrepancy scoring uses clipped alpha={alpha_clipped:.2e})")
    plt.tight_layout()
    fig.savefig(os.path.join(SANITY_CHECK_DIR, "sim_vs_target_ecg_first_guess.png"), dpi=150)
    plt.show()

def sanity_check_mesh_and_electrodes(xs, ys, zs, surface_info_field, electrodes_xyz):
    """Visual check: mesh voxel cloud + electrode positions together, so
    electrode placement can be judged relative to the actual heart geometry
    rather than by comparing raw coordinate magnitudes in the abstract."""
    import pyvista as pv

    mesh_points = np.column_stack([xs, ys, zs]).astype(float)
    cloud = pv.PolyData(mesh_points)
    cloud["surface_info_field"] = surface_info_field

    electrode_names = ["LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6"]
    electrode_colors = {
        "LA": "yellow", "RA": "cyan", "LL": "magenta", "RL": "brown",
        "V1": "darkred", "V2": "darkorange", "V3": "gold",
        "V4": "darkgreen", "V5": "darkblue", "V6": "indigo",
    }

    p = pv.Plotter()
    p.add_mesh(cloud, scalars="surface_info_field", cmap="RdBu_r",
               point_size=3, render_points_as_spheres=True, opacity=0.5,
               label="myocardium")

    for name, pos in zip(electrode_names, electrodes_xyz):
        p.add_mesh(pv.PolyData(np.array([pos])), color=electrode_colors[name],
                    point_size=18, render_points_as_spheres=True, label=name)

    p.add_legend()
    p.add_text("Sanity check: mesh + electrode positions\n"
               "(electrodes should sit outside the heart, at plausible "
               "torso-scale distances -- LA/RA near 'shoulders', "
               "LL/RL further 'below')", font_size=10)
    p.show(screenshot=os.path.join(SANITY_CHECK_DIR, "mesh_and_electrodes.png"))

    mesh_bounds = {
        "x": (xs.min(), xs.max()), "y": (ys.min(), ys.max()), "z": (zs.min(), zs.max())
    }
    print("Mesh bounds (um):")
    for axis, (lo, hi) in mesh_bounds.items():
        print(f"  {axis}: [{lo:.0f}, {hi:.0f}]  (span {hi - lo:.0f})")

    print("\nElectrode positions relative to mesh centroid (um):")
    mesh_centroid = mesh_points.mean(axis=0)
    for name, pos in zip(electrode_names, electrodes_xyz):
        offset = np.array(pos) - mesh_centroid
        dist = np.linalg.norm(offset)
        print(f"  {name}: offset=({offset[0]:.0f}, {offset[1]:.0f}, {offset[2]:.0f}), "
              f"distance from mesh centroid={dist:.0f} um ({dist/1000:.1f} mm)")

def isolate_qrs_window_shared(leads_targ_in, reference_lead, sampling_rate, padding_s=0.02,
                                slope_threshold_fraction=0.05, min_flat_duration_s=0.02):
    """Find QRS onset/offset directly via a slope-based method, rather than
    NeuroKit2's ecg_delineate (which needs a much longer multi-beat
    recording than a single ~350-sample representative beat provides).

    Method: the beat is already R-peak-centred at t=0. Starting from the
    R-peak, walk outward in each direction until the signal's local slope
    (rate of change) drops below a fraction of the peak QRS slope AND
    stays low for min_flat_duration_s -- i.e. until the trace flattens
    out into the isoelectric segment before P-wave / after QRS.
    """
    if reference_lead not in leads_targ_in:
        raise ValueError(f"Reference lead {reference_lead} not found in leads_targ_in")

    ref_times_s, ref_signal = leads_targ_in[reference_lead]
    r_idx = int(np.argmin(np.abs(ref_times_s)))  # index closest to t=0 (R-peak)

    dt = 1.0 / sampling_rate
    slope = np.gradient(ref_signal, dt)
    peak_slope = np.max(np.abs(slope[max(0, r_idx - 20):min(len(slope), r_idx + 20)]))
    slope_thresh = slope_threshold_fraction * peak_slope
    min_flat_samples = int(min_flat_duration_s * sampling_rate)

    def find_boundary(start_idx, direction):
        """Walk outward from start_idx in `direction` (+1 or -1) until
        slope stays below threshold for min_flat_samples consecutive points."""
        idx = start_idx
        flat_run = 0
        while 0 <= idx < len(slope):
            if abs(slope[idx]) < slope_thresh:
                flat_run += 1
                if flat_run >= min_flat_samples:
                    return idx - direction * (min_flat_samples - 1)
            else:
                flat_run = 0
            idx += direction
        return 0 if direction < 0 else len(slope) - 1

    onset_idx = find_boundary(r_idx, direction=-1)
    offset_idx = find_boundary(r_idx, direction=+1)

    per_lead_detections = {reference_lead: (onset_idx, offset_idx)}
    # for completeness/QC, run the same slope method on every lead independently
    for lead, (times_s, signal) in leads_targ_in.items():
        if lead == reference_lead:
            continue
        lead_r_idx = int(np.argmin(np.abs(times_s)))
        lead_slope = np.gradient(signal, dt)
        lead_peak_slope = np.max(np.abs(lead_slope[max(0, lead_r_idx-20):min(len(lead_slope), lead_r_idx+20)]))
        lead_thresh = slope_threshold_fraction * lead_peak_slope if lead_peak_slope > 0 else slope_thresh

        def find_boundary_lead(start_idx, direction, sl, thresh):
            idx, flat_run = start_idx, 0
            while 0 <= idx < len(sl):
                if abs(sl[idx]) < thresh:
                    flat_run += 1
                    if flat_run >= min_flat_samples:
                        return idx - direction * (min_flat_samples - 1)
                else:
                    flat_run = 0
                idx += direction
            return 0 if direction < 0 else len(sl) - 1

        lead_onset = find_boundary_lead(lead_r_idx, -1, lead_slope, lead_thresh)
        lead_offset = find_boundary_lead(lead_r_idx, +1, lead_slope, lead_thresh)
        per_lead_detections[lead] = (lead_onset, lead_offset)

    pad_samples = int(padding_s * sampling_rate)
    start = max(0, onset_idx - pad_samples)
    end = min(len(ref_times_s), offset_idx + pad_samples)

    leads_targ_in_trimmed = {}
    for lead, (times_s, signal) in leads_targ_in.items():
        trimmed_times = times_s[start:end] - times_s[start]
        trimmed_signal = signal[start:end]
        leads_targ_in_trimmed[lead] = [trimmed_times, trimmed_signal]

    window_info = {
        "shared_start_idx": start, "shared_end_idx": end,
        "reference_lead": reference_lead,
        "per_lead_detections": per_lead_detections,
    }
    return leads_targ_in_trimmed, window_info


def plot_qrs_isolation_qc(leads_targ_in_raw, window_info, sampling_rate):
    """Show the shared window applied (from reference_lead) against every
    lead's own RAW signal and INDEPENDENT delineation -- if independent
    onset/offset marks diverge noticeably from the shared window on
    non-reference leads, that's a sign leads aren't well time-aligned and
    per-lead delineation + realignment may be needed instead."""
    import matplotlib.pyplot as plt

    limb_leads = list(leads_targ_in_raw.keys())
    start_idx, end_idx = window_info["shared_start_idx"], window_info["shared_end_idx"]
    ref_lead = window_info["reference_lead"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, lead in zip(axes.flat, limb_leads):
        times_s, signal = leads_targ_in_raw[lead]
        ax.plot(times_s, signal, color="black", lw=1.0, label="raw signal")

        shared_start_t = times_s[start_idx] if start_idx < len(times_s) else None
        shared_end_t = times_s[end_idx - 1] if end_idx - 1 < len(times_s) else None
        if shared_start_t is not None and shared_end_t is not None:
            ax.axvspan(shared_start_t, shared_end_t, color="green", alpha=0.15,
                        label=f"shared window (from {ref_lead})")

        onset_idx, offset_idx = window_info["per_lead_detections"].get(lead, (None, None))
        if onset_idx is not None:
            ax.axvline(times_s[onset_idx], color="blue", lw=1.2, linestyle="--",
                        label="this lead's own onset" if lead != ref_lead else None)
        if offset_idx is not None:
            ax.axvline(times_s[offset_idx], color="red", lw=1.2, linestyle="--",
                        label="this lead's own offset" if lead != ref_lead else None)

        title = f"{lead}" + (" (reference)" if lead == ref_lead else "")
        ax.set_title(title)
        ax.legend(fontsize=6)

    fig.suptitle("QRS isolation QC: shared window (green) vs. each lead's own independent detection "
                 "(blue/red dashed)\nIf dashed lines fall far outside the green band on non-reference "
                 "leads, leads may not be well time-aligned -- consider per-lead delineation + realignment")
    plt.tight_layout()
    fig.savefig(os.path.join(SANITY_CHECK_DIR, "qrs_isolation_qc.png"), dpi=150)
    plt.show()

def sanity_check_target_leads(leads_targ_in):
    """Plot the representative-beat target leads being fed into the fit,
    so a loading/format bug (wrong column, wrong units, empty lead) is
    caught before it silently produces a bad discrepancy score."""
    import matplotlib.pyplot as plt

    limb_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, lead in zip(axes.flat, limb_leads):
        if lead not in leads_targ_in:
            ax.set_title(f"{lead} (MISSING)")
            continue
        times_s, signal = leads_targ_in[lead]
        ax.plot(times_s, signal, color="black", lw=1.2)
        ax.axvline(0, color="gray", lw=0.5, linestyle=":")
        ax.set_title(f"{lead}  (n={len(signal)}, range=[{signal.min():.3f}, {signal.max():.3f}])")
        ax.set_xlabel("time (s)")
    fig.suptitle("Sanity check: target ECG leads being fitted against")
    plt.tight_layout()
    fig.savefig(os.path.join(SANITY_CHECK_DIR, "target_leads_raw.png"), dpi=150)
    plt.show()

    print("Target lead summary:")
    for lead in limb_leads:
        if lead not in leads_targ_in:
            print(f"  {lead}: MISSING")
            continue
        times_s, signal = leads_targ_in[lead]
        print(f"  {lead}: {len(signal)} samples, "
              f"t=[{times_s[0]:.3f}, {times_s[-1]:.3f}]s, "
              f"amplitude range=[{signal.min():.3f}, {signal.max():.3f}]")

def save_best_sim_ecg_snapshot(iter_no, current_iter_params, all_electrodes, population_diff_scores,
                                 leads_target, times_s, times_target_s, out_dir):
    """Save the best-scoring simulated ECG at this iteration -- both as a
    plot (for quick visual convergence tracking) and as raw data (for
    later reanalysis/replotting without rerunning anything)."""
    best_key = min(population_diff_scores, key=population_diff_scores.get)
    best_score = population_diff_scores[best_key]

    leads_sim_best = ecg.ten_electrodes_to_twelve_leads(all_electrodes[best_key])
    limb_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]

    alpha = qrsm2.find_optimal_scaling(
        {l: leads_sim_best[l] for l in limb_leads},
        {l: leads_target[l] for l in limb_leads},
    )

    # unpack params into individually-shaped pieces -- can't savez a tuple
    # containing both a fixed-size (v_endo, v_myo) pair and a variable-
    # length root_indices sequence as one array
    (v_endo, v_myo), root_indices = current_iter_params[best_key]

    # --- save raw data, so this can be replotted/reanalysed later without rerunning the search ---
    np.savez(
        os.path.join(out_dir, f"iter_{iter_no:04d}_best_sim_ecg.npz"),
        iter_no=iter_no,
        best_score=best_score,
        alpha=alpha,
        v_endo=v_endo,
        v_myo=v_myo,
        root_indices=np.array(root_indices),
        **{f"sim_{l}": leads_sim_best[l] for l in limb_leads},
        **{f"target_{l}": leads_target[l] for l in limb_leads},
        times_s=times_s,
        times_target_s=times_target_s,
    )

    # --- save a quick-look plot ---
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, lead in zip(axes.flat, limb_leads):
        ax.plot(times_s, leads_sim_best[lead] * alpha, label="simulated (scaled)", color="red")
        ax.plot(times_target_s, leads_target[lead], label="target", color="black", alpha=0.6)
        ax.set_title(lead)
        ax.legend(fontsize=7)
    fig.suptitle(f"Iteration {iter_no}: best score={best_score:.5f}")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"iter_{iter_no:04d}_best_sim_ecg.png"), dpi=120)
    plt.close(fig)

    return best_score


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

    # --- Load real electrodes + target ECG ---
    electrodes_xyz = load_electrodes_xyz(ELECTRODE_PATH)
    leads_targ_in = load_leads_targ_in(REP_ECG_BEAT_DIR, limb_leads=("I", "II", "III", "aVR", "aVL", "aVF"))

    RUN_SANITY_CHECKS = False
    if RUN_SANITY_CHECKS:
        sanity_check_mesh_and_electrodes(xs, ys, zs, surface_info_field, electrodes_xyz)
        sanity_check_target_leads(leads_targ_in)

    leads_target, times_target_s, times_s, total_time_s = qrsm2.prepare_target_leads(
        leads_targ_in, ITER_DT_S, QRS_SAFETY_S
    )

    # --- Preprocessing for pseudo ECG computation ---
    grid_dict = alg_utils.make_grid_dictionary(xs, ys, zs)
    neighbour_arrays, *_ = ecg.get_neighbour_arrays(xs, ys, zs, DX, grid_dict)
    elec_grads = ecg.precompute_elec_grads(xs, ys, zs, electrodes_xyz, DX, neighbour_arrays).astype(np.float32)
    grid_endo_dict = alg_utils.make_grid_dictionary(xs_endo, ys_endo, zs_endo)

    root_nodes_neighbour_dist_um = 2 * ROOT_NODES_DIST_APART_UM
    candidate_root_points, candidate_root_neighbours = qrsm2.mesh_subset_with_dist_constraint(
        alg_endo, ROOT_NODES_DIST_APART_UM, root_nodes_neighbour_dist_um
    )

    # candidate_root_node_indices contains indices (corresponding to the
    # original alg mesh) of possible root nodes -- matches original script's
    # use of grid_endo_dict alongside grid_dict, even though `flag` itself
    # isn't used further downstream in the original either.
    candidate_root_node_indices = []
    flag = [0 for _ in range(len(xs_endo))]
    for point in candidate_root_points:
        flag[grid_endo_dict[point]] = 1
        candidate_root_node_indices.append(grid_dict[point])

    print(f"Candidate root node locations: {len(candidate_root_node_indices)}")

    adjacency_list_26 = ecg.compute_adjacency_displacement(xs, ys, zs, DX, grid_dict, NEIGHBOURS_26)

    v_endos = list(range(V_ENDO_MIN, V_ENDO_MAX + 1, V_ENDO_DIFF))
    v_myos = list(range(V_MYO_MIN, V_MYO_MAX + 1, V_MYO_DIFF))
    v_endos.sort()
    v_myos.sort()
    v_space = len(v_endos) * len(v_myos)
    print(f"Velocity search space: v_endo={v_endos}, v_myo={v_myos} ({v_space} combinations)")

    param_args = [
        (v_endo, v_myo, adjacency_list_26, endo_mask, USE_FIBERS, candidate_root_node_indices)
        for v_endo in v_endos for v_myo in v_myos
    ]
    batch_size = int(math.ceil(v_space / N_PROCESSORS))
    batched_param_args = list(qrsm2.batcher(param_args, batch_size))

    STOP_AFTER_FIRST_ITER_FOR_DIAGNOSTICS = False

    print(f"Precomputing {v_space} activation time matrices...")
    all_all_time_matrices = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_PROCESSORS) as executor:
        results = executor.map(qrsm2.compute_time_matrix_batch, batched_param_args)
        for batch_result in results:
            all_all_time_matrices.update(batch_result)

    print(f"Max QRS time simulated: {max(times_s) * 1000}ms")
    print(f"{len(candidate_root_points)} root node parameter space")

    # --- Initialise root nodes and conduction velocity parameters ---
    current_iter_params = qrsm2.init_roots_and_vels(
        N_TRIES, MIN_N_ROOT_NODES, MAX_N_ROOT_NODES, candidate_root_node_indices,
        v_endos, v_myos, params_best_guess=None, use_best_guess=False
    )

    mutated_params, all_ids_and_diff_scores = {}, {}
    runtimes, iter_median_scores = [], []

    best_ever_score = np.inf
    best_ever_electrodes = None
    best_ever_activation = None
    best_ever_params = None

    # --- Main iterative refinement of activation loop (faithful to original main_qrsmatch_revisit.py) ---
    for iter_no in range(N_ITERATIONS):
        n_tries = len(current_iter_params)
        n_per_batch = int(round(n_tries / N_PROCESSORS))
        n_per_batch = 1 if n_per_batch == 0 else n_per_batch
        tries = np.arange(n_tries)
        print(f"=== Iter {iter_no}: {n_tries} / {n_per_batch} ===", flush=True)

        all_electrodes, activation_times_s = qrsm2.batch_qrs_runner(
            n_tries, n_per_batch, qrsm2.pseudo_ecg_qrs, times_s, qrsm2.ap_heaviside,
            electrodes_xyz, elec_grads, DX, total_time_s, neighbour_arrays,
            all_all_time_matrices, current_iter_params
        )

        if iter_no == 0:
            first_try_key = list(current_iter_params.keys())[0]
            first_root_params = current_iter_params[first_try_key]
            print(f"\nFirst guess params: v_endo/v_myo={first_root_params[0]}, "
                  f"n_root_nodes={len(first_root_params[1])}, "
                  f"root_indices={first_root_params[1]}")
            print(f"Activation time range for first guess: "
                  f"{min(activation_times_s[first_try_key]) * 1000:.1f} - "
                  f"{max(activation_times_s[first_try_key]) * 1000:.1f} ms")

            if RUN_FIRST_ITER_DIAGNOSTICS:
                plot_activation_map(xs, ys, zs, activation_times_s[first_try_key])
                raw = all_electrodes[first_try_key]
                print(f"Raw electrode signal shape: {raw.shape}")
                for i, name in enumerate(["LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6"]):
                    print(f"  {name}: min={raw[i].min():.3e}, max={raw[i].max():.3e}, "
                          f"any_nonzero={np.any(raw[i] != 0)}")
                leads_sim_first = ecg.ten_electrodes_to_twelve_leads(all_electrodes[first_try_key])
                plot_sim_vs_target_ecg(leads_sim_first, leads_target, times_s, times_target_s)

            if STOP_AFTER_FIRST_ITER_FOR_DIAGNOSTICS:
                print("Stopping after first-guess diagnostics.")
                return

        # --- Reset per-iteration bookkeeping (matches original: fresh each iteration) ---
        population_ids_check, population_ids, ids_and_ecgs_ats_params = {}, {}, {}
        population_activation_times = {}

        for i_try in tries:
            param_id = qrsm2.hash_qrs_param(current_iter_params[i_try])
            population_ids[i_try] = param_id
            population_ids_check[param_id] = 1

        all_leads_sim, population_diff_scores = qrsm2.analyse_sim_qrs_leads(
            all_electrodes, leads_target, times_s, times_target_s, discrep_func
        )

        for i_try, params in current_iter_params.items():
            v_params, root_indices = params
            root_indices = list(root_indices)
            root_indices.sort()

            store_activation_times = np.round(np.array(activation_times_s[i_try]) * 1000)
            store_activation_times_ms = np.array(store_activation_times, dtype=np.uint16)
            population_activation_times[i_try] = store_activation_times_ms

            param_id = qrsm2.hash_qrs_param((v_params, tuple(root_indices)))
            all_ids_and_diff_scores[param_id] = [population_diff_scores[i_try], iter_no]
            ids_and_ecgs_ats_params[param_id] = [all_leads_sim[i_try], store_activation_times_ms, params]

        # --- Track best-ever result (addition on top of original logic, for
        # end-of-run visualisation -- doesn't affect the search itself) ---
        iter_best_key = min(population_diff_scores, key=population_diff_scores.get)
        iter_best_score = population_diff_scores[iter_best_key]
        if iter_best_score < best_ever_score:
            best_ever_score = iter_best_score
            best_ever_electrodes = all_electrodes[iter_best_key]
            best_ever_activation = activation_times_s[iter_best_key]
            best_ever_params = current_iter_params[iter_best_key]

        if iter_no % SAVE_SIM_ECG_EVERY_N_ITERS == 0:
            snap_score = save_best_sim_ecg_snapshot(
                iter_no, current_iter_params, all_electrodes, population_diff_scores,
                leads_target, times_s, times_target_s, CONVERGENCE_HISTORY_DIR
            )
            print(f"  [snapshot saved: iter {iter_no}, best score={snap_score:.5f}]")

        # --- Retrieval of diff scores in the population but not simulated
        # this iteration (RESTORED from original -- keeps previously-scored
        # good individuals in circulation rather than losing them once
        # they're no longer part of the freshly-simulated batch) ---
        population_params = current_iter_params.copy()
        new_key = max(population_diff_scores.keys()) + 1

        for key, params in mutated_params.items():  # mutated_params is of the population size
            params = params[0], tuple(params[1])
            param_id = qrsm2.hash_qrs_param(params)

            if param_id in all_ids_and_diff_scores and param_id not in population_ids_check:
                population_diff_scores[new_key] = all_ids_and_diff_scores[param_id][0]
                population_params[new_key] = params
                population_ids[new_key] = param_id
                population_ids_check[param_id] = 1
                new_key += 1

        print(f"len(population_diff_scores)={len(population_diff_scores)}")
        print(f"len(ids_and_ecgs_ats_params)={len(ids_and_ecgs_ats_params)}")

        scores = list(population_diff_scores.values())
        iter_median_scores.append(np.median(scores))

        keys_below, keys_above = ecg.tie_aware_proportional_split(population_diff_scores, 87.5)

        mutated_params = qrsm2.mutate_pop_params(
            keys_above, keys_below, population_params, alg, grid_dict,
            candidate_root_node_indices, candidate_root_neighbours, v_endos, v_myos,
            all_ids_and_diff_scores
        )

        if iter_no > 0:  # Retrieval of activation times from last iteration
            current_param_ids = {qrsm2.hash_qrs_param(params) for params in current_iter_params.values()}
            prev_population_ids = {param_id: local_index for local_index, param_id in population_ids.items()}
            next_index = max(population_activation_times.keys(), default=-1) + 1

            for key, params in population_params.items():
                param_id = qrsm2.hash_qrs_param(params)

                if param_id in current_param_ids:  # Skip if model already in this iter
                    continue
                if param_id in prev_population_ids:  # Use activation times from previous iter
                    prev_key = prev_population_ids[param_id]
                    if prev_key in population_activation_times:
                        population_activation_times[next_index] = population_activation_times[prev_key]
                        next_index += 1

        next_iter_params, next_iter_tries_ct = {}, 0
        for key, params in mutated_params.items():
            v_params, root_indices = params
            param_id = qrsm2.hash_qrs_param((v_params, tuple(root_indices)))
            if param_id not in all_ids_and_diff_scores:
                next_iter_params[next_iter_tries_ct] = params
                next_iter_tries_ct += 1

        current_iter_params = next_iter_params

        min_diff_score = min(population_diff_scores.values())
        print(f"Min score: {round(min_diff_score, 5)}, unique params tested: {len(all_ids_and_diff_scores)}, "
              f"next iter population: {len(current_iter_params)}")

        converged, _ = ecg.runtime_stop_condn(iter_no, iter_median_scores, window_size=5, stop_thresh=STOP_THRESH)
        if converged:
            print("Converged, stopping.")
            break

    # --- Final result: visualise the best fit found across the ENTIRE search ---
    print(f"\n=== FINAL RESULT ===")
    print(f"Best score: {best_ever_score:.5f}")
    print(f"Best params: v_endo/v_myo={best_ever_params[0]}, n_root_nodes={len(best_ever_params[1])}, "
          f"root_indices={best_ever_params[1]}")

    import pyvista as pv
    import datetime
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    points = np.column_stack([xs, ys, zs]).astype(float)
    cloud = pv.PolyData(points)
    cloud["activation_ms"] = np.asarray(best_ever_activation) * 1000
    p = pv.Plotter()
    p.add_mesh(cloud, scalars="activation_ms", cmap="plasma",
               point_size=4, render_points_as_spheres=True)
    p.add_text(f"FINAL best activation map\nscore={best_ever_score:.5f}, "
               f"range: {cloud['activation_ms'].min():.1f}-{cloud['activation_ms'].max():.1f} ms",
               font_size=10)
    p.show(screenshot=os.path.join(SANITY_CHECK_DIR, f"FINAL_activation_map_{run_timestamp}.png"))

    leads_sim_best = ecg.ten_electrodes_to_twelve_leads(best_ever_electrodes)
    alpha_final = qrsm2.find_optimal_scaling(
        {l: leads_sim_best[l] for l in ["I", "II", "III", "aVR", "aVL", "aVF"]},
        {l: leads_target[l] for l in ["I", "II", "III", "aVR", "aVL", "aVF"]},
    )
    print(f"Final clipped scaling factor: {alpha_final:.3e}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, lead in zip(axes.flat, ["I", "II", "III", "aVR", "aVL", "aVF"]):
        ax.plot(times_s, leads_sim_best[lead] * alpha_final, label="simulated (best fit)", color="red")
        ax.plot(times_target_s, leads_target[lead], label="target", color="black", alpha=0.6)
        ax.set_title(lead)
        ax.legend(fontsize=8)
    fig.suptitle(f"FINAL best fit: score={best_ever_score:.5f}")
    plt.tight_layout()
    fig.savefig(os.path.join(SANITY_CHECK_DIR, f"FINAL_sim_vs_target_ecg_{run_timestamp}.png"), dpi=150)
    plt.show()

    print(f"Done in {time.time() - runtime_start:.1f}s")


if __name__ == "__main__":
    main()
