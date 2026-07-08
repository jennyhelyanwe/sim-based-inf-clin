# Per-patient workflow: rate-dependent CV & repolarisation heterogeneity

**Goal:** for each patient, compare bulk conduction velocity and APD/repolarisation
heterogeneity across heart rates, using within-patient rate-contrasts (rather than
absolute resting values) as the between-group differentiator.

**Rationale:** resting QRS duration hasn't shown clear group separation, and most of
the observed ECG difference is in S-wave amplitude rather than timing — so a single
resting-state CV estimate may not carry the signal. CV and APD are both well-established
as rate-dependent (restitution behaviour), so using exercise heart rate as a physiological
perturbation may unmask group differences that are invisible at a single operating point.
This also turns the comparison into a within-patient paired design (Δ per patient,
rest→exercise), which is typically better powered than a cross-sectional comparison of
absolute values.

---

## Prerequisites (one-off, not per-patient)

1. **Mesh voxelisation** — done. `voxelise.py` produces the `.alg` output with
   `surface_info_field` (LV/RV endocardial labels), `trans` (transmural coordinate),
   and `apexb` (apex-base coordinate).
2. **Rate-specific AP lookup tables** — extend the Pythonised MATLAB AP-table generator
   to produce a *family* of tables, one per target pacing cycle length, spanning the
   resting → exercise heart rate range (e.g. every ~10–20 bpm). Each table keeps the
   existing `(APD90, APD50) -> [times, Vm]` structure; only the underlying single-cell
   model's paced CL changes.
3. **Electrode positions (LA/RA/LL)** — pending DICOM/torso-geometry estimation
   strategy. Same electrode positions are reused across all heart rates for a given
   patient (electrodes don't move with heart rate).
4. **ECG extraction** — resting 12-lead (limb-only) trace, plus one QRS+T-wave segment
   extracted per available exercise stage/heart rate, per patient.

---

## Per-patient steps

### Step 1 — Resting-state fit (establishes root nodes)
- Run the QRS-matching stage (`main_qrsmatch_revisit.py`,
  `discrepancy_name = "calc_discrepancy_limb_leads_only"`) on the resting ECG as
  normal: full search over root node combinations *and* `v_endo`/`v_myo`.
- Save the best root-node set and resting `v_endo`/`v_myo`. This root-node set is
  treated as fixed anatomical truth from here on (same conduction-system origin
  sites, independent of rate).

### Step 2 — Resting-state repolarisation fit
- Run the T-wave stage (`main_twavematch_revisit.py`) against the resting-rate AP
  table, using the resting activation times from Step 1.
- Save resting APD90 field / heterogeneity parameters (`apexb_param`, disk-based
  APD90 distribution).

### Step 3 — Per exercise heart rate: CV-only refit
For each available exercise stage:
- Set `use_best_guess = 1` in the QRS script, with `params_best_guess` set to the
  **Step 1 root nodes** (candidate root indices locked) — this restricts the search
  to `v_endo`/`v_myo` only, removing root-node vs. CV degeneracy from the comparison.
- Fit against that stage's extracted ECG.
- Record `v_endo`, `v_myo` for this heart rate.

### Step 4 — Per exercise heart rate: repolarisation refit
- Using the CV/activation-time result from Step 3, run the T-wave stage against the
  **rate-matched AP table** (nearest paced CL to that stage's actual heart rate) —
  not the resting-rate table.
- Record APD90 field / heterogeneity parameters for this heart rate.

### Step 5 — Within-patient contrasts
For each patient, compute:
- Δ`v_endo`, Δ`v_myo` (each exercise stage vs. rest)
- Δ APD90-heterogeneity metric (e.g. spread/skew of the disk-based APD90 field, or
  the review paper's chosen summary statistic) vs. rest
- Optionally, CV/APD *restitution slope* if ≥3 heart rate points are available per
  patient, rather than just a two-point difference.

### Step 6 — Between-group comparison
- Compare the **Δ metrics** (not absolute resting values) across patient groups.
- Sanity check first: confirm Δ metrics are stable across repeated fits with
  different `random_seed` at a given heart rate, before trusting between-group
  differences in the Δ — this is the same identifiability concern as at rest, just
  now applied to the contrast rather than the absolute value.

---

## Open dependencies before this can run end-to-end

- [ ] Rate-specific AP table generator (Pythonisation of MATLAB code, extended to a
      CL sweep)
- [ ] Electrode position estimation from DICOM/torso geometry
- [ ] Exercise-stage ECG extraction/trace quality check at higher heart rates
      (motion artifact risk)
- [ ] Decide the specific APD-heterogeneity summary statistic to track as "the"
      repolarisation-heterogeneity metric before running the full cohort, so
      Steps 5/6 aren't ad hoc after the fact

---

## Known limitations / caveats carried over from earlier discussion

- Only limb leads (I, II, III, aVR, aVL, aVF) are available — no precordial (V1-V6).
  This removes most spatial resolution over the anterior/septal wall, so root-node
  and APD-field localisation there will be more degenerate and more sensitive to the
  LA/RA/LL position estimate than a full 12-lead fit would be.
- `use_fibers = 0` by default in this pipeline: propagation is isotropic, so
  `v_endo`/`v_myo` are single bulk values per fit, not a spatial CV map. This
  workflow tracks rate-dependent *change* in these bulk values, not regional CV
  heterogeneity.
- Velocity search grid resolution is coarse (10 cm/s steps) — CV estimates (and
  their deltas) should be reported/interpreted at that resolution, not as continuous
  values.
- The BiV mesh is a single anatomical snapshot reused across all heart-rate fits for
  a given patient — a reasonable simplifying assumption, but ventricular geometry
  does shift somewhat with rate/loading in reality.
- The RV septum endo/epi convention for the `trans` field (feeding APD
  heterogeneity) is still an open question — pending confirmation from James
  Coleman on which convention the original benchmarks assumed.