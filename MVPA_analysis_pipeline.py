from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rsatoolbox
from rsatoolbox.vis import show_rdm
from collections import defaultdict
import nibabel as nib


from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import math_img, resample_to_img, load_img
from nilearn.masking import apply_mask
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting, image
from nilearn.plotting import plot_design_matrix

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.decomposition import PCA


########## PATH DIRECTORY ###########################################################################

sub_hector_path = Path(r"C:\Users\LENOVO P50\MVPA_IRMf\code_data\1_sujet_memoire\sub-hector")

anat = sub_hector_path / "anat"
fmap = sub_hector_path / "fmap"
func = sub_hector_path / "func"
level_1 = sub_hector_path / "level_1"
test_nilearn = sub_hector_path / "Test_nilearn"

########## PATH FILES ################################################################################

warsub = func / "warsub-061_task-drive_bold.nii"
spmT = level_1 / "spmT_0001.nii"
anat_img = anat / "sub-061_T1w.nii"

### BETA DIR ### 

beta_dir = sub_hector_path / "betas_fir_1"
beta_dir.mkdir(exist_ok=True)

results = sub_hector_path / "results"
results.mkdir(exist_ok = True)
n = 1
while True :
    results_dir = results / f"results_{n}"
    if not results_dir.exists(): 
        results_dir.mkdir()
        break 
    n += 1

print("[DEBUG] directorys created")
########## CREATION EVENT ##############################################################################

onsets = [
    40.627, 593.559, 1561.147, 1670.634,   # MAN
    50.627, 603.559, 1571.147, 1680.634,   # MAN_aON
    60.627, 613.559, 1581.147, 1690.634,   # HAD_aON
    555.949, 1519.780, 1616.864, 555.949,  # HAD
    570.949, 1534.780, 1631.864, 570.949,  # HAD_aOFF
    585.949, 1549.780, 1646.864, 585.949   # MAN_aOFF           # ATTENTION j'ai réutilisé le doublon pour le 4 ème RUN
]

durations = [
    10,10,10,10,
    10,10,10,10,
    10,10,10,10,
    15,15,15,15,
    15,15,15,15,
    15,15,15,15
]

trial_type = []

for cond, n in [
    ("MAN", 4),
    ("MAN_aON", 4),
    ("HAD_aON", 4),
    ("HAD", 4),
    ("HAD_aOFF", 4),
    ("MAN_aOFF", 4),
]:

    for i in range(n):
        trial_type.append(f"{cond}_{i+1}")

events = pd.DataFrame({
    "onset": onsets,
    "duration": durations,
    "trial_type": trial_type
})
print("[DEBUG] Events Head : ","\n", events.head())


# save file, maybe reuse later. 
events.to_csv(results / "events.tsv", sep="\t", index=False)


conditions = events["trial_type"].values
print("[DEBUG] Conditions (5 firsts) : ",conditions[0:5] , "...")


################## RECUPERATION OR CREATION BETA MAPS ############################### 


### CONSTANTS 

t_r = 1.4 
n_bins = 12 

# essaye de les récupérer, et si il récupère qqchse on considère que c'est bon, pas très robuste mais a toujours marché 

beta_files = sorted(beta_dir.glob("*.nii.gz"))
betas_exist = len(beta_files) > 0       # remplacer par égal run*delays*condition

print("[DEBUG] Existing beta files:", len(beta_files))

beta_condition = []

if betas_exist :
    print("[INFO] Loading existing betas...")

    beta_files = sorted(beta_dir.glob("*.nii.gz")) # on le remet, pas obligé, mais ça évite les erreurs 

    for f in beta_files:
        beta = load_img(f)
        beta_condition.append(beta)

    print("[DEBUG] Beta Loaded")

else :
    hrf_model = "fir"              # calcule la HRF dans range(n_bins), au lieu de l'importer 

    glm = FirstLevelModel(
        t_r = t_r,
        noise_model = "ar1",
        smoothing_fwhm = None,
        hrf_model = hrf_model,
        drift_model = "cosine",
        fir_delays=range(n_bins)
    )
    glm.fit(warsub, events)
    design_matrix = glm.design_matrices_[0]    # on pourrait l'afficher 
    design_matrix.to_csv(results / "Design_Matrix.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_design_matrix(design_matrix, axes=ax)

    fig.savefig(results / "Design_Matrix_plot.png", dpi=600, bbox_inches="tight")
    print("[DEBUG] Design matrix saved")

    plt.close(fig)
    print("[INFO] Computing Betas ...")

    fir_columns = [c for c in design_matrix.columns if "_delay" in c]     # à vérifier, supprimer les betas et faire des prints, (normalement c'est bon)
                                                                          # récupère toutes les colonnes, elles sont marquées delay celles qui nous intéresse. et ça ne prends pas les drift ou autre 
    for col in fir_columns:
        beta = glm.compute_contrast(col, output_type="effect_size")
        filename = beta_dir / f"{col}.nii.gz"
        beta.to_filename(filename)     # même chose que nibabel.save enregistre l'image au path et nom indiqué 
        beta_condition.append(beta)
    print("[DEBUG] Beta computed and loaded")

beta_files = sorted(beta_dir.glob("*.nii.gz"))     # On vient ensuite les rechercher depuis le disque 

def get_trial_and_delay(f):                        # Fonction utilitaire qui sert à renvoyer l'essai et le numera de delay, est utile pour trier, mais aussi pour accéder et vérifier.  
                                                   
    name = f.name
    name = name.replace(".nii.gz", "").replace(".nii", "")
    parts = name.split("_delay_")
    trial = parts[0]
    delay = int(parts[1])

    return (trial, delay)

beta_files = sorted(beta_files, key=get_trial_and_delay)

print("[DEBUG] Number Betas : ", len(beta_files))



################## DISTANCE TO INTERVAL [MAN;AUTO] #################################################################################



beta_dict = {}
anat_img = image.load_img(anat_img)
print("[DEBUG] Anatomical image loaded")

for f in beta_files:
    trial, delay = get_trial_and_delay(f)
    beta_dict[(trial, delay)] = load_img(f)


def compute_interval(start_img, end_img, mid_img):   
    low_img = math_img("np.minimum(img1, img2)", img1=start_img, img2=end_img)
    high_img = math_img("np.maximum(img1, img2)", img1=start_img, img2=end_img)
    
    dist_low = math_img("imgX - imgL", imgX=mid_img, imgL=low_img)
    dist_high = math_img("imgX - imgH", imgX=mid_img, imgH=high_img)

    result_img = math_img("np.where(abs(dlow) < abs(dhigh), dlow, dhigh)", dlow=dist_low, dhigh=dist_high) 
    result_img = resample_to_img(result_img, anat_img)

    return result_img

# exemple, mais plus tard il faudra faire quelquechose de plus complexe, qui teste plus de possibilitées 

man_img = beta_dict[("MAN_1", 11)]
had_img = beta_dict[("HAD_aON_1", 0)]
x_img   = beta_dict[("MAN_aON_1", 9)]

result_img = compute_interval(man_img, had_img, x_img)
print("[DEBUG] Interval Image calculated")
# sauvegarde et affichage 

nib.save(result_img, results_dir / "distance_map_delay18.nii")
print("[DEBUG] Interval Image saved")

plotting.plot_stat_map(result_img, bg_img=anat_img, cmap="Reds", threshold=0, colorbar=True, title="Distance map")
plotting.show()

# Quand plusieurs images, faire un système de nommage et enregistrement dans un dossier spécifique propre, si possible en .nii
# quand liste d'images, addition uniquement des valeurs positives 


######################### IMPORT ATLAS ################################################################################################# 



atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_img = atlas.maps
labels = atlas.labels

print("[DEBUG] Atlas loaded")
print("[DEBUG] All Labels in atlas :","\n", labels)


##########################################################################################################################################
##########################################################################################################################################
#########################      START PIPELINE ROI      ###################################################################################
##########################################################################################################################################

roi_list = ["Precentral Gyrus"]
print("[DEBUG] ROI list :", roi_list)

for roi in roi_list : 

    roi_path = results_dir / roi
    roi_path.mkdir(exist_ok=True)

    print("[DEBUG] directory ", roi," Created" )
############### MASKING ##################################

    roi_index = labels.index(roi)
    print("[DEBUG] ROI index:", roi_index)

    m1_mask = math_img(f"img == {roi_index}",img=atlas_img)

    reference_img = beta_condition[0]

    m1_mask = resample_to_img(m1_mask, reference_img, interpolation="nearest")

    X = apply_mask(beta_condition,m1_mask)
    print("[DEBUG] Shape Beta Masked : ", X.shape)

######################### RSA PIPELINE ################################ 










































### DATASET CREATION 
# peut être à mettre après events 

    defined_order = ["MAN", "MAN_aON", "HAD_aON", "HAD", "HAD_aOFF", "MAN_aOFF"]


    conditions = []
    delays = []
    runs = []
    trials = []

    for f in beta_files:

        trial, delay = get_trial_and_delay(f)

        if trial.startswith("MAN_aON"):
            cond = "MAN_aON"

        elif trial.startswith("HAD_aON"):
            cond = "HAD_aON"

        elif trial.startswith("HAD_aOFF"):
            cond = "HAD_aOFF"

        elif trial.startswith("MAN_aOFF"):
            cond = "MAN_aOFF"

        elif trial.startswith("MAN"):
            cond = "MAN"

        elif trial.startswith("HAD"):
            cond = "HAD"

        run = int(trial.split("_")[-1])

        conditions.append(cond)
        delays.append(delay)
        runs.append(run)
        trials.append(trial)

    metadata_df = pd.DataFrame({"condition": conditions, "delay": delays, "run": runs, "trial": trials})
    data_dataset = rsatoolbox.data.Dataset(measurements=X, obs_descriptors=metadata_df.to_dict("list"), channel_descriptors={"voxel": np.arange(X.shape[1])})

    print("[DEBUG] Dataset Per Delay head : ","\n", data_dataset.to_df().head())
    print("[DEBUG] Dataset Per Delay shape : ", data_dataset.to_df().shape)


# RDM PER DELAY (at the same time point, rdm so for 12 delays, an subplots with n_bins) cross-validated by runs. 


    rdm_list = []

    for d in range(n_bins):

        idx = metadata_df["delay"].values == d

        data_delay = rsatoolbox.data.Dataset(measurements=X[idx], obs_descriptors={"condition": metadata_df.loc[idx, "condition"].values, "run": metadata_df.loc[idx, "run"].values}, 
                                            channel_descriptors={"voxel": np.arange(X.shape[1])})

        rdm_delay = rsatoolbox.rdm.calc_rdm(data_delay, method="crossnobis", descriptor="condition", cv_descriptor="run")
        rdm_list.append(rdm_delay)

        print("[INFO] computed delay:", d)

    rdms_time = rsatoolbox.rdm.concat(rdm_list)
    print("[DEBUG] RDM Per Delay head :", "\n",rdms_time.to_df().head())

# plot 

    n_delays = n_bins

    rows = int(np.ceil(np.sqrt(n_delays)))
    cols = int(np.ceil(n_delays / rows))

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1]*cols + [0.05])

    axes = []

    for r in range(rows):
        for c in range(cols):
            axes.append(fig.add_subplot(gs[r, c]))

    axes = np.array(axes)
    all_mats = rdms_time.get_matrices()
    global_min = np.min(all_mats)
    global_max = np.max(all_mats)

    for i in range(n_delays):
        ax = axes[i]
        mat = all_mats[i]

        im = ax.imshow(mat, cmap="viridis", vmin=global_min, vmax=global_max)

        ax.set_title(f"Delay {i} ({i * t_r:.1f} s)", fontsize=9)

        ax.set_xticks(range(len(defined_order)))
        ax.set_xticklabels(defined_order, rotation=90, fontsize=7)

        ax.set_yticks(range(len(defined_order)))
        ax.set_yticklabels(defined_order, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Crossnobis distance", fontsize=11)
    fig.suptitle("Time-resolved RSA — Crossnobis distance", fontsize=13, y=1)
    fig.savefig(roi_path / f"RSA_by_delay_{roi}.png", dpi=600, bbox_inches="tight")
    print("[DEBUG] Fig RDM Per Delay Saved")
    plt.tight_layout()

    plt.show()


### ALL DELAYS FULL MATRIX (288, 288)  -> pas cross-nobis, on ne peut plus, on a plus les runs pour pouvoir faire le roulement 

# Building dataset 

    dataset_full = rsatoolbox.data.Dataset(measurements=X, obs_descriptors={"pattern": np.arange(len(X)), "run": metadata_df["run"].values, "condition": metadata_df["condition"].values, "delay": metadata_df["delay"].values},
        channel_descriptors={"voxel": np.arange(X.shape[1])})

    

    rdm_full = rsatoolbox.rdm.calc_rdm(dataset_full, method="euclidean", descriptor="pattern")

    matrix_full = rdm_full.get_matrices()[0]

    print("[DEBUG] Dataset All delays head : ","\n",dataset_full.to_df().head())
    print("[DEBUG] Dataset All delays shape : ","\n",dataset_full.to_df().shape)
    print("[DEBUG] RDM All delays RDM head : ","\n", rdm_full.to_df().head())
# PLOT 
    plt.figure(figsize=(12, 10))

    im = plt.imshow(matrix_full, cmap="viridis", interpolation="nearest")

    plt.colorbar(im, label="Euclidean distance")

    plt.title("FULL RDM — 288 × 288 Euclidean")
    plt.savefig(roi_path / f"RSA_ALL_delay_{roi}.png", dpi=600, bbox_inches="tight")
    print("[DEBUG] RDM All delays saved")
    plt.tight_layout()
    plt.show()


### MEAN by run 

# Building dataset 

    patterns = []
    pattern_labels = []
    pattern_runs = []

    for cond in defined_order:
        for d in range(n_bins):
            idx = ((metadata_df["condition"] == cond) & (metadata_df["delay"] == d))
            if idx.sum() == 0: continue
            patterns.append(X[idx])
            label = f"{cond}_d{d}"
            pattern_labels.extend([label] * idx.sum())
            pattern_runs.extend(metadata_df.loc[idx, "run"].values)

    patterns = np.vstack(patterns)

    dataset_cond_delay = rsatoolbox.data.Dataset(measurements=patterns, obs_descriptors={"pattern": pattern_labels, "run": pattern_runs}, channel_descriptors={"voxel": np.arange(patterns.shape[1])})

    print("[DEBUG] Dataset condition × delay head :","\n",dataset_cond_delay.to_df().head())
    print("[DEBUG] Dataset condition × delay shape :","\n",dataset_cond_delay.to_df().shape)

# RDM 

    rdm_cond_delay = rsatoolbox.rdm.calc_rdm(dataset_cond_delay, method="crossnobis", descriptor="pattern", cv_descriptor="run")
    matrix = rdm_cond_delay.get_matrices()[0]

    print("[DEBUG] RDM condition × delay shape : ", rdm_cond_delay.to_df().shape)

# axis label : 
    labels = []

    for cond in defined_order:

        for d in range(n_bins):

            labels.append(
                f"{cond}_d{d}"
            )

# PLOT 

    plt.figure(figsize=(12, 10))

    im = plt.imshow(matrix, cmap="viridis")
    plt.colorbar(im, label="Crossnobis distance")

    tick_positions = np.arange(0, len(labels), n_bins)

    plt.xticks(tick_positions, defined_order, rotation=90)
    plt.yticks(tick_positions, defined_order)

    plt.title("RDM — Condition × Delay")
    plt.savefig(roi_path / f"RSA_CONDITION_DELAY_{roi}.png", dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


### FULL MEAN cond x cond 


# building dataset 

    patterns = []
    pattern_conditions = []
    pattern_runs = []

    for cond in defined_order:
        for r in sorted(metadata_df["run"].unique()):
            idx = ((metadata_df["condition"] == cond) & (metadata_df["run"] == r))

            if idx.sum() == 0:
                continue

            mean_pattern = X[idx].mean(axis=0)
            patterns.append(mean_pattern)
            pattern_conditions.append(cond)
            pattern_runs.append(r)

    patterns = np.vstack(patterns)

# dataset 
    
    dataset_cond = rsatoolbox.data.Dataset(measurements=patterns, obs_descriptors={"condition": pattern_conditions, "run": pattern_runs},channel_descriptors={"voxel":np.arange(patterns.shape[1])})

    print("[DEBUG] Dataset condition head :", dataset_cond.to_df().head())
    print("[DEBUG] Dataset condition shape :", dataset_cond.to_df().shape)

# RDM 

    print("[INFO] Computing 6 × 6 condition RDM")

    rdm_cond = rsatoolbox.rdm.calc_rdm(dataset_cond, method="crossnobis", descriptor="condition", cv_descriptor="run")

    matrix_cond = rdm_cond.get_matrices()[0]

    print("[DEBUG] condition RDM head :", rdm_cond.to_df().head())

# PLOT 

    plt.figure(figsize=(7, 6))

    im = plt.imshow(matrix_cond, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, label="Crossnobis distance")

    plt.xticks(range(len(defined_order)), defined_order,rotation=90)
    plt.yticks(range(len(defined_order)), defined_order)

    plt.title("RDM — Conditions (6 × 6, Crossnobis)")
    plt.savefig(roi_path / f"RSA_CONDITION_{roi}.png", dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


################## MODEL DECODER ##################################################################################################################################################

# Building dataset 
    print("[START] MODEL PIPELINE")
    conditions_list = []

    for f in beta_files:
        trial, delay = get_trial_and_delay(f)
        conditions_list.append(trial)

    conditions = np.array(conditions_list)

    print("[DEBUG] CONDITIONS SHAPE:", conditions.shape)

    train_conditions = [
        "MAN",
        "HAD_aON",
        "HAD",
        "MAN_aOFF"
    ]

    train_mask = np.array([any(cond.startswith(tc) for tc in train_conditions)for cond in conditions])

    X_train = X[train_mask]
    cond_train = conditions[train_mask]

    print("[DEBUG] X_train shape (stable states): ", X_train.shape)

    def train_label(cond):
        if cond.startswith("MAN"):
            return 1
        if cond.startswith("MAN_aOFF"):
            return 1
        if cond.startswith("HAD"):
            return 0
        if cond.startswith("HAD_aON"):
            return 0

    y_train = np.array([train_label(c)for c in cond_train])

    print("[DEBUG] y_train shape :", y_train.shape)


# model 

    clf = Pipeline([("scaler", StandardScaler()), ("feature_selection", SelectKBest(f_classif, k=500)), ("svc", SVC(kernel="linear"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    mean_cv = scores.mean()

    print("[DEBUG] Cross-validation scores:","\n", scores)
    print("[DEBUG] Mean CV accuracy : " ,mean_cv)

    clf.fit(X_train, y_train)

# Prediction on transition 

    test_conditions = [
        "MAN_aON",
        "HAD_aOFF"
    ]

    test_mask = np.array([any(cond.startswith(tc) for tc in test_conditions) for cond in conditions])
    X_test = X[test_mask]
    cond_test = conditions[test_mask]

    y_pred = clf.predict(X_test)

    mask_MAN_aON = np.array([c.startswith("MAN_aON")for c in cond_test])
    mask_HAD_aOFF = np.array([c.startswith("HAD_aOFF") for c in cond_test])

    pred_MAN_aON = y_pred[mask_MAN_aON]
    pred_HAD_aOFF = y_pred[mask_HAD_aOFF]

    print("Predictions MAN_aON :","\n" , pred_MAN_aON)
    print("Predictions HAD_aOFF :", "\n", pred_HAD_aOFF)

# Plot 

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(pred_MAN_aON, marker="o")

    axes[0].set_title(f"MAN_aON predictions\nCV accuracy = {mean_cv:.2f}")

    axes[0].set_ylim(-0.1, 1.1)

    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Prediction (0=AUTO, 1=MANUAL)")


    axes[1].plot(pred_HAD_aOFF, marker="o")

    axes[1].set_title(f"HAD_aOFF predictions\nCV accuracy = {mean_cv:.2f}")

    axes[1].set_ylim(-0.1, 1.1)

    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Prediction (0=AUTO, 1=MANUAL)")

    plt.savefig(roi_path / f"Decodeur_plot_transition_pred_{roi}.png", dpi=600, bbox_inches="tight")
    print("[DEBUG] model prediction Figure saved")
    plt.tight_layout()
    plt.show()


######################### PCA PIPELINE ########################################################################################################################################


# numbers PC choice, explained variance treshold = 0.9 
    max_components = min(20, X.shape[0], X.shape[1])

    pca_full = PCA(n_components=max_components )

    X_pca_full = pca_full.fit_transform(X)


    explained = pca_full.explained_variance_ratio_

    cumulative = np.cumsum(explained)

    components = np.arange(1, len(explained) + 1)

    n_components = 0

    for i in range(len(components)):
        if cumulative[i]> 0.9 :
            print("[DEBUG] NUMBERS PCA : ",components[i])
            n_components = components[i]
            break
    
    print("[DEBUG] PC numbers over 0.90 : ", n_components)

    plt.figure(figsize=(8, 5))

    plt.plot(components, explained, marker="o", label="Explained variance")
    plt.plot(components, cumulative, marker="o", label="Cumulative variance")
    plt.plot(components[n_components-1], cumulative[n_components-1], marker="x", c="red", label = f"value 1st componant > 0.9 = {n_components}")

    plt.axhline(0.9, linestyle="--")
    plt.xlabel("Number of components")
    plt.ylabel("Variance explained")

    plt.title("Explained variance vs number of components")
    plt.savefig(roi_path / f"PCA_choice_number_pc_{roi}.png", dpi=600, bbox_inches="tight")
    print("[DEBUG] PCA numbers components choice fig saved")
    plt.legend()
    plt.show()




# Parameters 

    TR = 1.4
    window = 45

    group_1 = [
        "MAN",
        "MAN_aON",
        "HAD_aON"
    ]

    group_2 = [
        "HAD",
        "HAD_aOFF",
        "MAN_aOFF"
    ]


# Process Onsets 

    cond_to_onset = {}

    for cond in defined_order:
        first_trial = f"{cond}_1"
        onset_value = events.loc[events["trial_type"] == first_trial,"onset"].values[0]
        cond_to_onset[cond] = onset_value

    print("[DEBUG] cond_to_onset:", cond_to_onset)

# PCA 

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)

# Process Offsets (like a jitter)

    def compute_run_offsets(anchor_condition):

        offsets = {}

        for run in metadata_df["run"].unique():
            trial_name = f"{anchor_condition}_{run}"
            onset = events.loc[events["trial_type"] == trial_name,"onset"].values[0]  

            remainder = onset % TR

            if remainder == 0:
                offset = 0
            else:
                offset = TR - remainder

            offsets[run] = offset
            print(f"[DEBUG] Run {run}", "onset =", onset, "offset =", offset )

        return offsets

# Build real timeline (onsets + offsets and after iterate for each beta) -> this gives abscisse to plot PCA 

    def build_aligned_time(anchor_condition):

        offsets = compute_run_offsets(anchor_condition)
        anchor_onset = cond_to_onset[anchor_condition]

        times = []

        for i in range(len(metadata_df)):
            run = metadata_df.iloc[i]["run"]
            delay = metadata_df.iloc[i]["delay"]
            condition = metadata_df.iloc[i]["condition"]
            offset = offsets[run]

            condition_shift = (cond_to_onset[condition]- anchor_onset)
            t = delay * TR + offset + condition_shift
            times.append(t)

        return np.array(times)

# Plot all the runs 

    def plot_group_subTR_stacked(groups, titles, anchor_conditions):

        fig, axes = plt.subplots(len(groups), 1, figsize=(14, 8), sharex=True)

        for idx in range(len(groups)):

            group = groups[idx]
            title = titles[idx]
            anchor_condition = anchor_conditions[idx]

            t_local = build_aligned_time(anchor_condition)
            mask = metadata_df["condition"].isin(group)

            Xg = X_pca[mask]
            tg = t_local[mask]
            cg = metadata_df.loc[mask, "condition"].values

            keep = (tg >= 0) & (tg <= window)

            Xg = Xg[keep]
            tg = tg[keep]
            cg = cg[keep]

            order = np.argsort(tg)

            Xg = Xg[order]
            tg = tg[order]

            ax = axes[idx]

            for pc in range(n_components):
                ax.plot(tg, Xg[:, pc], marker="o", linewidth=1.5, label=f"PC{pc+1}")

            anchor_onset = cond_to_onset[anchor_condition]

            for c in group:

                rel_time = (cond_to_onset[c] - anchor_onset)

                if 0 <= rel_time <= window:
                    ax.axvline(rel_time, linestyle="--", alpha=0.4)
                    ax.text(rel_time, 0, c, rotation=90, fontsize=9, verticalalignment="bottom")

            ax.set_ylabel("PCA activity")
            ax.set_title(title)

        axes[-1].set_xlabel("Time after onset (s)")

        axes[0].legend()
        plt.savefig(roi_path / f"PCA_ALL_run_{roi}.png", dpi=600, bbox_inches="tight")
        plt.tight_layout()
        plt.show()


    plot_group_subTR_stacked(
        [group_1, group_2],
        ["MAN aligned", "HAD aligned"],
        ["MAN", "HAD"]
    )

    def plot_group_subTR_runs_stacked(group, title, record,anchor_condition):

        t_local = build_aligned_time(anchor_condition)

        runs_unique = sorted(metadata_df["run"].unique())

        fig, axes = plt.subplots(len(runs_unique), 1, figsize=(14, 10), sharex=True)

        for idx in range(len(runs_unique)):

            run = runs_unique[idx]

            mask = (
                metadata_df["condition"].isin(group)
                & (metadata_df["run"] == run)
            )

            Xg = X_pca[mask]
            tg = t_local[mask]

            keep = (tg >= 0) & (tg <= window)

            Xg = Xg[keep]
            tg = tg[keep]

            order = np.argsort(tg)

            Xg = Xg[order]
            tg = tg[order]

            ax = axes[idx]

            for pc in range(n_components):
                ax.plot(tg, Xg[:, pc], marker="o", linewidth=1.5)

            anchor_onset = cond_to_onset[anchor_condition]

            for c in group:

                rel_time = (cond_to_onset[c] - anchor_onset)

                if 0 <= rel_time <= window:
                    ax.axvline(rel_time, linestyle="--", alpha=0.4)
                    ax.text(rel_time, 0, c, rotation=90, fontsize=9, verticalalignment="bottom")

            ax.set_ylabel(f"Run {run}")

        axes[-1].set_xlabel("Time after onset (s)")

        fig.suptitle(title)
        plt.savefig(roi_path / f"PCA_EACH_run_{record}_{roi}.png", dpi=600, bbox_inches="tight")
        plt.tight_layout()
        plt.show()


    plot_group_subTR_runs_stacked(
        group_1,
        "Sub-TR PCA dynamics — MAN aligned",
        'MAN_ALIGNED',
        anchor_condition="MAN"
    )

    plot_group_subTR_runs_stacked(
        group_2,
        "Sub-TR PCA dynamics — HAD aligned",
        "HAD_ALIGNED",
        anchor_condition="HAD"
    )



















































































































































