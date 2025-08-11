from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import io as sio
from skimage import metrics as skim
from skimage import io as skio
import matplotlib.pyplot as plt
import argparse
import pandas as pd


#Helper functions ==========================================

def _to_int_labels(arr: np.ndarray) -> np.ndarray:
    """Ensure label maps are `int64` (handle floats/NaNs gracefully)."""
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.nan_to_num(arr, nan=0.0).astype(np.int64, copy=False)
    return arr

def _comb2(x: np.ndarray | int) -> np.ndarray | int:
    """n choose 2, element‑wise for arrays (integer arithmetic)."""
    return x * (x - 1) // 2

# All of the metric calculations =========================
#VOI, ARI, 

def _variation_of_information(seg: np.ndarray, gt: np.ndarray) -> float:
    seg_i, gt_i = _to_int_labels(seg), _to_int_labels(gt)
    split, merge = skim.variation_of_information(gt_i, seg_i)
    return split + merge

def _probabilistic_rand_index(seg: np.ndarray, gt: np.ndarray) -> float:
    """True (probabilistic) Rand Index ∈ [0,1], higher is better."""
    # print(str(seg.shape) + " seg shape")
    # print(str(gt.shape) + " gt shape")
    seg_flat = _to_int_labels(seg).ravel()
    gt_flat = _to_int_labels(gt).ravel()
    N = seg_flat.size
    _, seg_enc = np.unique(seg_flat, return_inverse=True)
    _, gt_enc = np.unique(gt_flat, return_inverse=True)
    n_seg = seg_enc.max() + 1
    n_gt = gt_enc.max() + 1

    idx = seg_enc * n_gt + gt_enc
    cont = np.bincount(idx, minlength=n_seg * n_gt).reshape((n_seg, n_gt))

    ### Adjsuted RI
    sum_comb_nij = _comb2(cont).sum(dtype=np.int64)
    sum_comb_row = _comb2(cont.sum(axis=1)).sum(dtype=np.int64)
    sum_comb_col = _comb2(cont.sum(axis=0)).sum(dtype=np.int64)
    total_pairs  = _comb2(N)

    if total_pairs == 0:
        return 1.0   # single-pixel “segmentations” are identical

    # --- ARI formula ---------------------------------------------------------
    # prod = (sum_comb_row * sum_comb_col) / total_pairs
    prod = sum_comb_row * (sum_comb_col / total_pairs)

    numerator   = sum_comb_nij - prod
    denominator = 0.5 * (sum_comb_row + sum_comb_col) - prod

    # Degenerate case: both partitions have only one cluster
    return 1.0 if denominator == 0 else float(numerator / denominator)

# file loading functions ===========================================

def _load_mat_segs(mat_path: Path) -> Tuple[List[np.ndarray], List[int]]:
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    segs_raw = data["segs"]
    eigs_raw = data["eigs"]

    # Matlab file loader flattens arrays with one dim
    # Fixing that - issues when there is only one segmentation
    # But setting squeeze_me to false breaks downstream stuff
    # print(segs_raw)
    if isinstance(segs_raw, np.ndarray) and segs_raw.ndim != 1: 
        segs_raw_new = np.empty(1, dtype=object)
        segs_raw_new[0] = segs_raw
        segs_raw = segs_raw_new

    # Flatten MATLAB cell → Python list
    if isinstance(segs_raw, np.ndarray) and segs_raw.ndim == 0:
        segs_list = [segs_raw.item()]
    elif isinstance(segs_raw, np.ndarray):
        segs_list = list(segs_raw)
    else:
        segs_list = list(segs_raw) if isinstance(segs_raw, (list, tuple)) else [segs_raw]

    segs = [_to_int_labels(np.asarray(s)) for s in segs_list]
    eigs = (list(eigs_raw) if isinstance(eigs_raw, (np.ndarray, list))
            else [int(eigs_raw)])

    assert len(segs) == len(eigs)
    return segs, eigs


def _load_gt(gt_path: Path) -> List[np.ndarray]:
    data = sio.loadmat(gt_path, squeeze_me=True, struct_as_record=False)
    gts_raw = data["groundTruth"]
    # print("ground truth shape:")
    # print(gts_raw.shape)
    if isinstance(gts_raw, np.ndarray) and gts_raw.ndim == 0:
        gts_raw = [gts_raw.item()]
    else:
        gts_raw = list(gts_raw) if isinstance(gts_raw, np.ndarray) else gts_raw
    return [_to_int_labels(np.asarray(g.Segmentation)) for g in gts_raw]

# Methods to evaluate BSDS ==============================================

def analyse_one_result(pred_mat: Path, gt_mat: Path
                       ) -> Tuple[List[float], List[float], List[int]]:
    # print(pred_mat)
    segs, eig_nums = _load_mat_segs(pred_mat)
    gts = _load_gt(gt_mat)
    ris, vois = [], []
    for seg in segs:
        ri_vals = [_probabilistic_rand_index(seg, gt) for gt in gts]
        voi_vals = [_variation_of_information(seg, gt) for gt in gts]
        ris.append(float(np.mean(ri_vals)))
        vois.append(float(np.mean(voi_vals)))
    return ris, vois, eig_nums


def analyse_bsds_results(split: str = "test") -> None:
    out_csv =  Path("results/bsds/csv_results/" +  experiment_name + ".csv")
    
    gt_dir_expanded = (gt_root / split).expanduser()
    seg_dir_expanded = seg_dir.expanduser()
    if not seg_dir_expanded.is_dir():
        raise FileNotFoundError(seg_dir_expanded)
    if not gt_dir_expanded.is_dir():
        raise FileNotFoundError(gt_dir_expanded)
    

    # Get segmentation stats from the last generated stats file
    stats_df = pd.read_csv("results/bsds/csv_results/experimentStats.csv")
    # stats_df = stats_df.set_index('image')
    # print(stats_df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "k", "eigs", "ari", "voi", "duration", "graphSize", "averageDegree"])
        for pred_path in sorted(seg_dir_expanded.glob("*.mat")):
            img_id = pred_path.stem
            gt_path = gt_dir_expanded / f"{img_id}.mat"
            if not gt_path.exists():
                print(f"[warn] No ground truth for {img_id}, skipping …")
                continue


            # Get analysis stats 
            ris, vois, eigs = analyse_one_result(pred_path, gt_path)
            matching_rows = stats_df.loc[stats_df['image'] == int(img_id), ['duration', 'graphSize', 'averageDegree']]
            
            if not matching_rows.empty:
                print("Found runtime duration for image")
                row = matching_rows.iloc[0]
                runtime_duration = row['duration']
                graph_size = row['graphSize']
                avg_deg = row['averageDegree']
            else:
                runtime_duration = 0
                graph_size = 0
                avg_deg = 0

            # if img_id in stats_df.index:
            #     runtime_duration = stats_df.at[img_id, 'duration']
            # else: 
            #     runtime_duration = 0

            # Saving k as the highest num of eigenvectors
            for ri, voi, eig in zip(ris, vois, eigs):
                writer.writerow([
                    img_id,
                    eigs[-1],                  
                    eig,                    
                    f"{ri:.6f}",                     
                    f"{voi:.6f}", 
                    runtime_duration,
                    graph_size,
                    avg_deg              
                ])
    print(f" [Evaluation saved] to {out_csv}")






### Ground truth visualisations of the segmentations 
def export_groundtruth_visualisation(img_id) -> None:
    output_dir = Path("results/bsds/visualisations/ground_truth")
    gt_dir = gt_root.expanduser()
    out_dir = output_dir.expanduser()

    split_priority = ("test", "train")
    for split in split_priority:
        gt_path = gt_root / split / f"{img_id}.mat"
        if gt_path.exists():
            break
    else:
        raise FileNotFoundError("Ground‑truth not found in given splits")

    gts = _load_gt(gt_path)
    if not gts:
        print(f"[warn] {gt_path} is empty – skipping.")
    else: 
        # Generate the plots
        num_segs = len(gts)
        plt.figure(figsize= (4 * num_segs, 4))
        plt.suptitle("Ground truth segmentations for image " + img_id)
        for i in range(num_segs):
            plt.subplot(1, num_segs, i+1)
            plt.axis("off")

            gt_seg = gts[i]
            plt.imshow(gt_seg, cmap="tab20", interpolation="nearest")

        save_path = out_dir / f"{img_id}_groundtruth.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f" [saved - Ground Truth] {_friendly_save_msg(save_path)}")

    # plt.subplot(1, ncols, 1)
    # plt.imshow(image)
    # plt.title("Original")
    # plt.axis("off")

    # plt.subplot(1, ncols, 2)
    # plt.imshow(segs[idx_best_ri], interpolation="nearest", cmap="tab20")
    # plt.title("Best Clustering\n" + 
    #           str(eig_nums[idx_best_ri]) + " eigenvalues; "
    #           + str(eig_nums[-1]) + " clusters" 
    #         #   + "\nRand Index: " + str( f"{ri_scores[idx_best_ri]:.6f}"))
    #         + "\nARI: " + str( f"{ri_scores[idx_best_ri]:.6f}"))
    # plt.axis("off")

    # plt.subplot(1, ncols, 3)
    # plt.imshow(segs[idx_most_eigs], interpolation="nearest", cmap="tab20")
    # plt.title("Most Eigenvalues\n" + 
    #           str(eig_nums[idx_most_eigs]) + " eigenvalues; "
    #           + str(eig_nums[-1]) + " clusters" 
    #         #   + "\nRand Index: " + str( f"{ri_scores[idx_most_eigs]:.6f}"))
    #         + "\nARI: " + str( f"{ri_scores[idx_most_eigs]:.6f}"))
    # plt.axis("off")



    
    # # There are 5 different ground truth segmentations
    # segmentation_id = 4 # pick the segmentation at this id if the segmentation file contains multiple ones

    # gt_seg = gts[segmentation_id]

    # plt.figure(figsize=(4, 4))
    # plt.imshow(gt_seg, cmap="tab20", interpolation="nearest")
    # plt.axis("off")
    # plt.title("Ground Truth: " + img_id + ".png")

    # save_path = out_dir / f"{img_id}_groundtruth.png"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.close()

    # print(f" [saved - Ground Truth] {_friendly_save_msg(save_path)}")


#Methods to create the visualisations-------------------------------

def _friendly_save_msg(save_path: Path) -> str:
    #Return a readable path (relative if possible, else absolute).
    try:
        rel = save_path.resolve().relative_to(Path.cwd().resolve())
        return str(rel)
    except ValueError:
        return str(save_path.resolve())


def compare_segmentations(img_id: str,
                          split_priority: tuple[str, ...] = ("test", "train"),
                          save_path: Path | None = None,
                          show: bool = True) -> None:
    pred_file = seg_dir / f"{img_id}.mat"
    if not pred_file.exists():
        raise FileNotFoundError(pred_file)

    for split in split_priority:
        img_path = img_root / split / f"{img_id}.jpg"
        if img_path.exists():
            break
    else:
        raise FileNotFoundError("Image not found in given splits")

    for split in split_priority:
        gt_path = gt_root / split / f"{img_id}.mat"
        if gt_path.exists():
            break
    else:
        raise FileNotFoundError("Ground‑truth not found in given splits")

    image = skio.imread(img_path)
    segs, eig_nums = _load_mat_segs(pred_file)
    gts = _load_gt(gt_path)

    # Find segmentation with highest RI and most eigenvalues

    ri_scores = [
        float(np.mean([_probabilistic_rand_index(seg, gt) for gt in gts]))
        for seg in segs
    ]

    idx_best_ri = int(np.argmax(ri_scores))
    # idx_most_eigs = int(np.argmax(eig_nums)) 

    # Generate the plot
    # ncols = 1 + len(segs)
    ncols = 2


    plt.figure(figsize=(4 * ncols, 4))
    plt.suptitle("Results of experiment " + experiment_name )

    plt.subplot(1, ncols, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, ncols, 2)
    plt.imshow(segs[idx_best_ri], interpolation="nearest", cmap="tab20")
    plt.title(str(eig_nums[idx_best_ri]) + " eigenvalues; "
              + str(eig_nums[-1]) + " clusters" 
            #   + "\nRand Index: " + str( f"{ri_scores[idx_best_ri]:.6f}"))
            + "\nARI: " + str( f"{ri_scores[idx_best_ri]:.6f}"))
    plt.axis("off")

    # plt.subplot(1, ncols, 3)
    # plt.imshow(segs[idx_most_eigs], interpolation="nearest", cmap="tab20")
    # plt.title("Most Eigenvalues\n" + 
    #           str(eig_nums[idx_most_eigs]) + " eigenvalues; "
    #           + str(eig_nums[-1]) + " clusters" 
    #         #   + "\nRand Index: " + str( f"{ri_scores[idx_most_eigs]:.6f}"))
    #         + "\nARI: " + str( f"{ri_scores[idx_most_eigs]:.6f}"))
    # plt.axis("off")




    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" [saved] {_friendly_save_msg(save_path)}")

    if show:
        plt.show()
    else:
        plt.close()


def export_visualisations(seg_dir: Path = Path("results/bsds/segs"),
                        img_root: Path = Path("data/bsds/BSR/BSDS500/data/images"),
                        out_dir: Path = Path("results/bsds/visualisations"),
                        split_priority: tuple[str, ...] = ("test", "train")) -> None:
    seg_dir = seg_dir.expanduser()
    out_dir = out_dir.expanduser()
    if not seg_dir.is_dir():
        raise FileNotFoundError(seg_dir)

    for pred_path in sorted(seg_dir.glob("*.mat")):
        img_id = pred_path.stem
        save_path = out_dir / experiment_name / f"{img_id}.png"
        compare_segmentations(img_id,
                              split_priority=split_priority,
                              save_path=save_path,
                              show=False)
        #Generate ground truth visualisation 
        export_groundtruth_visualisation(img_id)
    print(f" [Visualisations saved] to {_friendly_save_msg(out_dir / experiment_name)}")


def parse_args():
    parser = argparse.ArgumentParser(description='Analyse BSDS Experiments')
    parser.add_argument('experiment_name', type=str, 
                        help="Experiment code name for naming purposes as defined in experiment_configurations.yaml")

    return parser.parse_args()


def main():
    args = parse_args()
    global experiment_name 
    global seg_dir
    global gt_root 
    global img_root
    experiment_name = args.experiment_name
    seg_dir = Path("results/bsds/segs")
    gt_root = Path("data/bsds/BSR/BSDS500/data/groundTruth")
    img_root = Path("data/bsds/BSR/BSDS500/data/images")

    print("Analysing BSDS experiment named: " + experiment_name)
    analyse_bsds_results()
    export_visualisations()

if __name__ == "__main__":
    main()
