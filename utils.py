import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold, StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def load_and_preprocess_data():
    """Loading and preprocessing the data

    Returns: Dataframes
    """
    uniqcounts = pd.read_csv("data/uniqcounts.tsv", sep="\t")
    samplemap = pd.read_csv("data/samplemap.tsv", sep="\t")

    sig = pd.read_csv("data/celltype_signatures_added_missingHV3.tsv", sep="\t")
    ratio = pd.read_csv("data/100kb_binned_SLratio.tsv", sep="\t")
    cov = pd.read_csv("data/100kb_binned_coverage.tsv", sep="\t")
    # set index and replace nan by 0
    uniqcounts = uniqcounts.set_index("length").T.fillna(0)
    uniqcounts.index.names = ["sample"]
    samplemap.set_index("sample", inplace=True)
    sig.set_index("sample", inplace=True)
    # change format
    sig = sig.pivot(columns="region-set", values="Dip area").merge(
        sig.pivot(columns="region-set", values="Dip depth"),
        right_index=True,
        left_index=True,
        suffixes=("_area", "_depth"),
    )
    # drop samples with dropout == yes
    samplemap = samplemap[samplemap.dropout != "yes"]
    uniqcounts = uniqcounts.loc[samplemap.index]
    sig = sig.loc[samplemap.index]
    # cov and ratio
    cov["chr"] = (
        cov["c1"] + "_" + cov["start"].astype(str) + "_" + cov["end"].astype(str)
    )
    cov = cov.drop(columns=["c1", "start", "end"]).set_index("chr").fillna(0).T
    ratio["chr"] = (
        ratio["c1"] + "_" + ratio["start"].astype(str) + "_" + ratio["end"].astype(str)
    )
    ratio = ratio.drop(columns=["c1", "start", "end"]).set_index("chr").fillna(0).T
    cov = cov.loc[samplemap.index]
    ratio = ratio.loc[samplemap.index]
    uniqcounts.name = "fragment length"
    sig.name = "cell signatures"
    ratio.name = "ratio"
    cov.name = "coverage"
    return uniqcounts, samplemap, sig, ratio, cov


def train_test(samplemap, include_hn=True):
    """
    Split into train and test

    Args:
        samplemap: dataframe with sample info
        include_hn (bool, optional): Whether to include the HN patients into train or not. Defaults to True.

    Returns: dataframes of train and test samples
    """
    if include_hn:
        train_samples = samplemap[samplemap.day == 0].copy()
        test_samples = samplemap[samplemap.day != 0].copy()
    else:
        train_samples = samplemap[(samplemap.day == 0) & (samplemap.group != "HN")]
        test_samples = samplemap[(samplemap.day != 0) | (samplemap.group == "HN")]
    train_samples["labels"] = [
        1 if elem == "Healthy" else 0 for elem in train_samples.group
    ]
    test_samples["labels"] = [
        1 if elem in [90, 180, 365] else 0 for elem in test_samples.day
    ]
    return train_samples, test_samples


def train_test_custom(
    samplemap,
    train_indices=[
        "HN1-BL",
        "HN2-BL",
        "HN3-BL",
        "HN4-BL",
        "HN5-BL",
        "HN6-BL",
        "HN7-BL",
        "HV1",
        "HV3",
        "HV4",
        "HV5",
        "HV6",
        "HV7",
        "HV8",
        "OMD1-BL",
        "OMD3-BL",
        "OMD4-BL",
        "OMD5-BL",
        "OMD6-BL",
        "OMD7-BL",
        "OMD8-BL",
        "OMD9-BL",
        "PV1",
        "PV3",
        "PV4",
        "PV5",
        "PV6",
        "PV7",
        "PV8",
        "PV9",
        "HN1-d180",
        "HN3-d180",
        "HN4-d180",
        "HN5-d180",
    ],
):
    """Split into train and test by providing the list of train samples

    Args:
        samplemap (_type_): df with sample info
        train_indices (list, optional): list of train samples
    Returns:
        dataframes of train and test samples
    """
    test_indices = [elem for elem in samplemap.index if elem not in train_indices]
    train_samples = samplemap.loc[train_indices]
    test_samples = samplemap.loc[test_indices]
    # the two next lines must be completed/adapted if you pick different train/test samples
    train_samples["labels"] = [
        1
        if index
        in [
            "HV1",
            "HV3",
            "HV4",
            "HV5",
            "HV6",
            "HV7",
            "HV8",
            "HN1-d180",
            "HN3-d180",
            "HN4-d180",
            "HN5-d180",
        ]
        else 0
        for index in train_samples.index
    ]
    test_samples["labels"] = [
        1 if elem in [90, 180, 365] else 0 for elem in test_samples.day
    ]
    return train_samples, test_samples


def refit_strategy_loo(cv_results):
    df = pd.DataFrame(cv_results)
    all_conf_matrices = {}
    F1_scores = np.empty(len(df.index))
    # very unstable way to fetch the number of splits,
    # but didnt really know how to do it otherwise without complicating
    # the code too much, since this function is only allowed to take
    # cv_results as input
    n_splits = len(df.filter(regex=r"(split.*_tn)").columns)
    print(f"number of splits {n_splits}")
    for index in df.index:
        conf_matrix = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for split in range(n_splits):
            for key in conf_matrix:
                column_name = "split" + str(split) + "_test_" + key
                conf_matrix[key] += df.loc[index, column_name]
        all_conf_matrices[index] = conf_matrix
        F1_scores[index] = (
            2
            * conf_matrix["tp"]
            / (2 * conf_matrix["tp"] + conf_matrix["fp"] + conf_matrix["fn"])
        )
    return np.argmax(F1_scores)


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return {"tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}


# def refit_strategy_kfold(cv_results):
#     """Custom strategy to pick best model during GridSearchCV
#     We chose the model with  overall best precision, recall and f1

#     Args:
#         cv_results (_type_): cv_results_ attribute from GridSearchCV

#     Returns:
#        int : index of best model in results dataframe
#     """
#     return (
#         pd.DataFrame(cv_results)[
#             ["rank_test_f1_macro", "rank_test_recall", "rank_test_precision"]
#         ]
#         .sum(axis=1)
#         .argmin()
#     )


def pipeline(
    features: pd.DataFrame,
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    search_strategy="llo",
    parameters={"C": [0.1, 1, 5, 6, 7, 8, 9, 10, 20, 30, 40, 100]},
) -> list:

    """Whole train, tune, refit, and plot pipeline

    Args:
        features (pd.DataFrame): features
        train_samples (pd.DataFrame): train samples
        test_samples (pd.DataFrame): test samples
        search_strategy (str): CV strategy. Defaults to "llo"
        parameters (dict, optional): Dictionary of parameters to try out during the GridsearchCV. Defaults to {"C": [0.1, 1, 5, 6, 7, 8, 9, 10, 20, 30, 40, 100]}.

    Returns:
        list: Names of features with non-zero coefficients
    """
    seed = 0
    # select train and test features
    train_features = features.loc[train_samples.index]
    test_features = features.loc[test_samples.index]
    print(f"Initially using {len(train_features.columns)} features")
    # convert to numpy array
    X_train = train_features.values
    y_train = train_samples.labels.values
    X_test = test_features.values
    y_test = test_samples.labels.values
    # scale train and test
    scaler = StandardScaler().fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    # shuffle training features
    X_train_shuffled, y_train_shuffled = shuffle(
        X_train_scaled, y_train, random_state=seed
    )
    # Find optimal parameters on train using grid search cv
    # clf = GridSearchCV(LogisticRegression(penalty = 'l1', max_iter = 10000, solver = 'liblinear', class_weight = 'balanced'), param_grid=parameters, cv = StratifiedKFold(n_splits = 10), scoring = 'f1_macro')
    if search_strategy == "llo":
        clf = GridSearchCV(
            LogisticRegression(
                penalty="l1",
                max_iter=10000,
                solver="liblinear",
                class_weight="balanced",
            ),
            param_grid=parameters,
            cv=LeaveOneOut(),
            refit=refit_strategy_loo,
            scoring=confusion_matrix_scorer,
        )
    else:
        clf = GridSearchCV(
            LogisticRegression(
                penalty="l1",
                max_iter=10000,
                solver="liblinear",
                class_weight="balanced",
            ),
            param_grid=parameters,
            cv=StratifiedKFold(n_splits=5),
            scoring="f1_macro",
        )

    clf.fit(X_train_shuffled, y_train_shuffled)
    # best algorithm (already refitted on whole train set)
    best = clf.best_estimator_
    print(f"best parameters {clf.best_params_}")
    print(f"Number of nonzero coeffs {len(best.coef_.squeeze().nonzero()[0])}")
    print(f"Kept features {train_features.columns[best.coef_.squeeze().nonzero()[0]]}")
    # compute metrics on train
    y_pred_train = best.predict(X_train_shuffled)
    print(f"Metrics on train")
    print(classification_report(y_train_shuffled, y_pred_train))
    # predict on test
    print(f"Metrics on test")
    y_pred_test = best.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_test))
    # probabilities on test and train
    # make the plots
    for X, samples, title in zip(
        (X_test_scaled, X_train_scaled),
        (test_samples, train_samples),
        ("Test samples", "Train samples"),
    ):
        prob_healthy = best.predict_proba(X)[:, 1]
        predicted_probas = samples.copy()
        predicted_probas["proba_healthy"] = np.round_(prob_healthy, 2)
        plt.figure(figsize=(20, 10))
        plt.bar(predicted_probas.index, predicted_probas.proba_healthy)
        _ = plt.xticks(rotation=90)
        plt.ylabel("Probability of healthy")
        plt.title(f"Using {features.name}, {title} subset")
    return train_features.columns[best.coef_.squeeze().nonzero()[0]]

def pipeline_sequ_selection(features, train_samples, test_samples, forward=True):
    seed = 0
    # select train and test features
    train_features = features.loc[train_samples.index]
    test_features = features.loc[test_samples.index]
    print(f"Initially using {len(train_features.columns)} features")
    # convert to numpy array
    X_train = train_features.values
    y_train = train_samples.labels.values
    X_test = test_features.values
    y_test = test_samples.labels.values
    # scale train and test
    scaler = StandardScaler().fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    # shuffle training features
    X_train_shuffled, y_train_shuffled = shuffle(
        X_train_scaled, y_train, random_state=seed
    )
    # algorithm to perform feature selection (and also final model fitting in our case)
    alg = LogisticRegression(
        max_iter=10000, solver="liblinear", class_weight="balanced"
    )
    
    direction = "forward" if forward else "backward"
    # greedy feature selection
    sfs = SequentialFeatureSelector(
        alg,
        scoring="f1_macro",
        direction=direction,
        cv=StratifiedKFold(n_splits = 7),
        n_features_to_select="auto",
        tol=0.001,
    )
    # finf the features and transform to get new X, containg only the selected features
    X_train_reduced = sfs.fit_transform(X_train_shuffled, y_train_shuffled)
    X_test_reduced = sfs.transform(X_test_scaled)
    print(
        f"keeping {len(sfs.get_feature_names_out(features.columns))} features : {sfs.get_feature_names_out(features.columns)}"
    )
    # train our model using the selected features
    best = alg.fit(X_train_reduced, y_train_shuffled)
    # compute metrics on train
    y_pred_train = best.predict(X_train_reduced)
    print(f"Metrics on train")
    print(classification_report(y_train_shuffled, y_pred_train))
    # predict on test
    print(f"Metrics on test")
    y_pred_test = best.predict(X_test_reduced)
    print(classification_report(y_test, y_pred_test))
    # probabilities on test and train
    # make the plots
    for X, samples, title in zip(
        (X_test_reduced, X_train_reduced),
        (test_samples, train_samples),
        ("Test samples", "Train samples"),
    ):
        prob_healthy = best.predict_proba(X)[:, 1]
        predicted_probas = samples.copy()
        predicted_probas["proba_healthy"] = np.round_(prob_healthy, 2)
        plt.figure(figsize=(20, 10))
        plt.bar(predicted_probas.index, predicted_probas.proba_healthy)
        _ = plt.xticks(rotation=90)
        plt.ylabel("Probability of healthy")
        plt.title(f"Using {features.name}, {title} subset")

def concat_features(df_list):
    return pd.concat(df_list, axis=1)


def get_embeddings(uniqcounts, ratio, cov, sig):
    """compute some embeddings"""
    algo_umap = umap.UMAP(random_state=42)
    algo_tsne = TSNE(n_components=2, random_state=42)
    algo_pca = PCA(n_components=2, random_state=42)
    embed_cov = pd.DataFrame(algo_umap.fit_transform(cov), index=cov.index)
    embed_ratio = pd.DataFrame(algo_umap.fit_transform(ratio), index=ratio.index)
    embed_uniq = pd.DataFrame(
        algo_umap.fit_transform(uniqcounts), index=uniqcounts.index
    )
    embed_sig = pd.DataFrame(algo_umap.fit_transform(sig), index=sig.index)
    embed_cov_tsne = pd.DataFrame(algo_tsne.fit_transform(cov), index=cov.index)
    embed_ratio_tsne = pd.DataFrame(algo_tsne.fit_transform(ratio), index=ratio.index)
    embed_uniq_tsne = pd.DataFrame(
        algo_tsne.fit_transform(uniqcounts), index=uniqcounts.index
    )
    embed_sig_tsne = pd.DataFrame(algo_tsne.fit_transform(sig), index=sig.index)
    embed_cov_pca = pd.DataFrame(algo_pca.fit_transform(cov), index=cov.index)
    embed_ratio_pca = pd.DataFrame(algo_pca.fit_transform(ratio), index=ratio.index)
    embed_uniq_pca = pd.DataFrame(
        algo_pca.fit_transform(uniqcounts), index=uniqcounts.index
    )
    embed_sig_pca = pd.DataFrame(algo_pca.fit_transform(sig), index=sig.index)

    return (
        embed_cov,
        embed_ratio,
        embed_uniq,
        embed_cov_tsne,
        embed_ratio_tsne,
        embed_uniq_tsne,
        embed_cov_pca,
        embed_ratio_pca,
        embed_uniq_pca,
        embed_ratio_pca,
        embed_uniq_pca,
        embed_sig_pca,
    )
