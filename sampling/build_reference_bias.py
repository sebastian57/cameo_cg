#!/usr/bin/env python3
"""Build a smooth TICA acquisition bias from reference data only.

Generalised from SAMPLING/tica_regional_weighting/build_reference_bias.py, which
hardcoded the 5-bead ala2 paths and 300 K. Bead count comes from the TICA artifact;
temperature must match the atomistic reference (298 K for the current ala2 setup).

    python -m sampling.build_reference_bias \
        --grid-dir  SAMPLING/tica_regional_weighting/results/ala2_bb6_reference \
        --reference local_work/input_data/ala2_cg_backbone_cb_6bead_200k.npz \
        --temperature 298.0 --lambda-value 0.25
"""
from __future__ import annotations
import argparse,hashlib,json,pickle,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path: sys.path.insert(0, str(_REPO))
from sampling.biases.tica_regional import SmoothTICABias, TICAProjection

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", type=Path, required=True, help="dir holding tica_projection_and_grid.npz + tica_model.pkl")
    p.add_argument("--reference", type=Path, required=True, help="reference NPZ for the force audit")
    p.add_argument("--temperature", type=float, required=True, help="K; MUST match the atomistic reference")
    p.add_argument("--lambda-value", type=float, default=0.25)
    p.add_argument("--mode", choices=("log_ratio", "transition_attractor"),
                   default="log_ratio",
                   help="log_ratio (default) is the -kT log(target/p_ref) form, which is "
                        "anti-density by construction and cannot be redirected by "
                        "reweighting `exploration`. transition_attractor places wells "
                        "directly on committor/transition cells with NO p_ref term, so "
                        "basin ratios are left to the reference data.")
    p.add_argument("--attractor-field", type=str, default=None, metavar="NPZ:KEY[+KEY..]",
                   help="use an OBSERVED per-cell field as the attractor target instead of "
                        "the committor-derived transition_component. Keys are summed, so "
                        "'tmap.npz:passage_frequency_beta_to_alphaL+passage_frequency_alphaL_to_beta' "
                        "targets both directions of one channel. Measured 2026-08-05: the "
                        "committor model correlates only +0.05..+0.53 with observed passage "
                        "frequency, so it points the bias at corridors the trajectory never used.")
    p.add_argument("--extra-support", default=None,
                   help="NPZ[+NPZ..] of CG coordinates whose occupied TICA cells JOIN the "
                        "KDE centre set. Fixes the support gap that no amount of reweighting "
                        "could: the reference has ~2 frames in the alphaL corridor, so there "
                        "were no centres there to carry attractor weight. Coordinate-only "
                        "campaigns qualify -- a bias field needs geometry, not forces.")
    p.add_argument("--attractor-coords", default=None,
                   help="NPZ[+NPZ..] of CG coordinates whose per-cell density is folded into "
                        "the attractor target, combined with the existing field by elementwise "
                        "max after scaling each to 1.")
    p.add_argument("--attractor-depth", type=float, default=2.0,
                   help="well depth A in kcal/mol for --mode transition_attractor")
    p.add_argument("--transition-weight", type=float, default=None,
                   help="rebuild the exploration term as (1-w)*sparsity + w*transition. "
                        "The stored default blends both; raising w concentrates the bias "
                        "on committor/transition cells rather than merely low-density "
                        "ones. Low density IS the free energy -- flattening it overwrites "
                        "the physics the model must learn, whereas transition frames are "
                        "safe because the mean force there points outward into the basins.")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def main():
    args=parse_args()
    BASE=args.grid_dir.resolve(); REFERENCE=args.reference.resolve()
    LAMBDA=float(args.lambda_value); KBT=0.00198720425864083*float(args.temperature)
    OUT=args.out or BASE/("smooth_reference_bias_lambda%s.npz"%str(LAMBDA).replace(".","p"))
    SUMMARY=BASE/"reference_only_bias_summary.json"

    with np.load(BASE/"tica_projection_and_grid.npz") as data:
        pairs=np.asarray(data["pair_indices"],dtype=np.int64)
        xedges,yedges=np.asarray(data["xedges"]),np.asarray(data["yedges"])
        counts=np.asarray(data["counts"],dtype=np.int64)
        eligible=np.asarray(data["eligible"],dtype=bool)
        p_ref=np.asarray(data["p_ref"],dtype=np.float64)
        sparsity=np.asarray(data["sparsity_component"],dtype=np.float64)
        transition=np.asarray(data["transition_component"],dtype=np.float64)
        exploration=np.asarray(data["exploration"],dtype=np.float64)
        target=np.asarray(data["pi_lambda_%s"%(("%g"%LAMBDA))],dtype=np.float64)
        committor=np.asarray(data["committor"],dtype=np.float64)
    if args.transition_weight is not None:
        w=float(args.transition_weight)
        if not 0.0<=w<=1.0: raise SystemExit("--transition-weight must be in [0,1]")
        exploration=(1.0-w)*sparsity+w*transition
        exploration=exploration/exploration.sum()
        target=(1.0-LAMBDA)*p_ref+LAMBDA*exploration
        target=target/target.sum()
        print(f"exploration rebuilt: {100*(1-w):g}% sparsity + {100*w:g}% transition")
    expected=(1.0-LAMBDA)*p_ref+LAMBDA*exploration
    expected=expected/expected.sum()
    np.testing.assert_allclose(target,expected,rtol=0,atol=1e-12)
    if np.any(exploration[~eligible]!=0): raise ValueError("Exploration mass outside reference eligibility")
    with (BASE/"tica_model.pkl").open("rb") as handle: model=pickle.load(handle)
    mean=np.asarray(model.mean_0,dtype=np.float64)
    coefficients=np.asarray(model.instantaneous_coefficients[:,:2],dtype=np.float64)
    projection=TICAProjection(pairs,mean,coefficients)
    nx,ny=len(xedges)-1,len(yedges)-1
    xc,yc=0.5*(xedges[:-1]+xedges[1:]),0.5*(yedges[:-1]+yedges[1:])
    occupied=counts>0
    # ---- support extension -------------------------------------------------------
    # The KDE centres are REFERENCE-occupied cells, and that is precisely why the alphaL
    # corridor could never be opened: the reference has ~2 frames there, so there are no
    # centres to carry attractor weight and per-channel reweighting changed nothing
    # (measured 2026-08-05: 1.856 vs 1.790, no better). Fixing it needs new CENTRES, not
    # new weights. Coordinate-only campaigns are enough for this -- a bias field needs
    # geometry, not forces -- which is what makes the alphaL exit/shooting runs usable
    # here despite having been written with nstfout=0.
    extra_counts=np.zeros_like(counts)
    if args.extra_support:
        for src in args.extra_support.split("+"):
            with np.load(src) as fd:
                key="R" if "R" in fd.files else "coords"
                Rx=np.asarray(fd[key],dtype=np.float64)
            zx=projection.transform(Rx)
            h,_,_=np.histogram2d(zx[:,0],zx[:,1],bins=[xedges,yedges]); h=h.ravel()
            new=int(((h>0)&(counts==0)&(extra_counts==0)).sum())
            extra_counts+=h.astype(np.int64)
            print(f"extra support {Path(src).name}: {len(Rx)} frames, {new} cells the reference never occupied")
        occupied=occupied|(extra_counts>0)
        print(f"centres: {int((counts>0).sum())} reference -> {int(occupied.sum())} extended")
    ij=np.column_stack(np.unravel_index(np.flatnonzero(occupied),(nx,ny)))
    centers=np.column_stack((xc[ij[:,0]],yc[ij[:,1]]))
    # p_ref is exactly 0 in the newly added cells. That is harmless in transition_attractor
    # mode -- tica_regional.tica_energy_gradient short-circuits to the attractor branch and
    # never reads reference_weights -- but the log_ratio path would take log(0).
    ref_weights=p_ref[occupied].copy()
    if ref_weights.sum()<=0: raise SystemExit("no reference mass on the centre set")
    ref_weights/=ref_weights.sum()
    target_weights=target[occupied].copy(); target_weights/=target_weights.sum()
    cell_width=np.asarray([np.diff(xc).mean(),np.diff(yc).mean()])
    bandwidth=1.25*cell_width
    bounds=np.asarray([[xedges[0],xedges[-1]],[yedges[0],yedges[-1]]])
    wall_k=0.04/cell_width**2
    provisional=SmoothTICABias(projection,centers,ref_weights,target_weights,bandwidth,KBT,bounds,wall_k)
    # Offset over REFERENCE-occupied centres only: extended cells have zero reference
    # weight, and including them drives the log-ratio energy to -inf.
    ref_occ_mask=(p_ref[occupied]>0)
    occupied_energy=np.asarray([provisional.tica_energy_gradient(z)[0] for z in centers[ref_occ_mask]])
    offset=-float(occupied_energy.min())
    bias=SmoothTICABias(projection,centers,ref_weights,target_weights,bandwidth,KBT,bounds,wall_k,offset)
    with np.load(REFERENCE) as data: reference_R=np.asarray(data["R"],dtype=np.float64)
    attractor_kwargs={}
    if args.mode=="transition_attractor":
        # weights on the SAME occupied centres the KDE already uses, so the smoothing
        # and the grid stay identical to the log-ratio artifact
        if args.attractor_field:
            src,keys=args.attractor_field.split(":",1)
            with np.load(src) as fd:
                field=sum(np.asarray(fd[k],dtype=np.float64).ravel() for k in keys.split("+"))
            if field.size!=p_ref.size:
                raise SystemExit(f"field has {field.size} cells, grid has {p_ref.size}")
            print(f"attractor field: {keys} from {src} "
                  f"({int((field>0).sum())} non-zero cells)")
            transition=field.reshape(p_ref.shape)
        if args.attractor_coords:
            # Per-cell passage density of coordinate-only corridor campaigns, folded into
            # the attractor target. Combined by ELEMENTWISE MAX after scaling each part to
            # max 1, so a cell is attractive if EITHER channel uses it disproportionately
            # and neither channel's absolute scale can swamp the other. This is a design
            # choice, not a derivation: R_TP (beta<->alphaR) and a raw passage density
            # (alphaL) are different quantities and there is no principled common
            # normalisation. Raw density is the right form for alphaL specifically because
            # rho_eq ~ 0 there -- dividing by it, as R_TP does, would be the anti-density
            # trap that made the original log-ratio bias useless.
            extra=np.zeros_like(p_ref)
            for src in args.attractor_coords.split("+"):
                with np.load(src) as fd:
                    key="R" if "R" in fd.files else "coords"
                    Rx=np.asarray(fd[key],dtype=np.float64)
                zx=projection.transform(Rx)
                h,_,_=np.histogram2d(zx[:,0],zx[:,1],bins=[xedges,yedges]); h=h.ravel()
                extra+=h
                print(f"attractor coords {Path(src).name}: {len(Rx)} frames over {int((h>0).sum())} cells")
            t=transition.astype(np.float64); e=extra.astype(np.float64)
            if t.max()>0: t=t/t.max()
            if e.max()>0: e=e/e.max()
            transition=np.maximum(t,e)
            print(f"attractor target = max(R_TP_scaled, corridor_density_scaled): "
                  f"{int((transition>0).sum())} non-zero cells")
        aw=transition[occupied].astype(np.float64)
        if aw.sum()<=0: raise SystemExit("transition_component is empty on occupied cells")
        aw=aw/aw.sum()
        probe=SmoothTICABias(projection,centers,ref_weights,target_weights,bandwidth,KBT,bounds,wall_k)
        rho=np.array([np.exp(SmoothTICABias._log_density_and_gradient(z,centers,aw,bandwidth)[0]) for z in centers])
        attractor_kwargs=dict(attractor_weights=aw,attractor_depth=float(args.attractor_depth),
                              attractor_norm=float(rho.max()))
        print(f"transition_attractor: depth {args.attractor_depth} kcal/mol, "
              f"rho_max {rho.max():.4g} over {len(centers)} centres")
    np.savez_compressed(OUT,n_beads=int(reference_R.shape[1]),pair_indices=pairs,**attractor_kwargs,tica_mean=mean,tica_coefficients=coefficients,centers=centers,reference_weights=ref_weights,target_weights=target_weights,bandwidth=bandwidth,kbt_kcal_mol=KBT,bounds=bounds,xedges=xedges,yedges=yedges,wall_k_kcal_mol=wall_k,energy_offset_kcal_mol=offset,lambda_value=LAMBDA,reference_counts=counts,committor=committor,eligible=eligible,reference_sparsity_component=sparsity,reference_transition_component=transition,reference_exploration=exploration,provenance="reference-only: occupancy+sparsity+committor-transition; no CG trajectory or ML model")
    audit=reference_R[np.linspace(0,len(reference_R)-1,3000,dtype=int)]
    max_force=[]; energy=[]
    for structure in audit:
        value,force,_=bias.evaluate_A(structure); energy.append(value); max_force.append(np.linalg.norm(force,axis=1).max())
    max_force=np.asarray(max_force); energy=np.asarray(energy)
    full_centers=np.stack(np.meshgrid(xc,yc,indexing="ij"),axis=-1).reshape(-1,2)
    bias_map=np.asarray([bias.tica_energy_gradient(z)[0] for z in full_centers])
    maps=[("Reference probability",p_ref),("Reference sparsity",sparsity),("Reference transition relevance",transition),("Reference exploration",exploration),(f"Target lambda={LAMBDA:g}",target),("Smooth bias (kcal/mol)",bias_map)]
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),constrained_layout=True)
    extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]
    for ax,(title,values) in zip(axes.flat,maps):
        image=ax.imshow(values.reshape(nx,ny).T,origin="lower",extent=extent,aspect="auto",cmap="turbo"); fig.colorbar(image,ax=ax); ax.set_title(title); ax.set_xlabel("TIC 1"); ax.set_ylabel("TIC 2")
    plot=BASE/"reference_only_bias_components.png"; fig.savefig(plot,dpi=220); plt.close(fig)
    summary={"status":"reference_only_bias_ready","artifact":str(OUT.resolve()),"sha256":sha256(OUT),"lambda":LAMBDA,"mode":args.mode,"transition_weight":args.transition_weight,"n_beads":int(reference_R.shape[1]),"target":f"{100*(1-LAMBDA):g}% reference occupancy + {100*LAMBDA:g}% reference sparsity/committor exploration","uses_cg_or_ml_data":False,"eligible_cells":int(eligible.sum()),"occupied_cells":int(occupied.sum()),"normalization":{"p_ref":float(p_ref.sum()),"exploration":float(exploration.sum()),"target":float(target.sum())},"force_audit_kcal_mol_A":{"frames":len(audit),"median":float(np.median(max_force)),"p95":float(np.quantile(max_force,.95)),"p99":float(np.quantile(max_force,.99)),"maximum":float(max_force.max())},"energy_audit_kcal_mol":{"minimum":float(energy.min()),"maximum":float(energy.max())},"plot":str(plot.resolve())}
    SUMMARY.write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
