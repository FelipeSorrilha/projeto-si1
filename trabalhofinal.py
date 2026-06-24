import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

#caminho base, tem que por o caminho do dataset aqui
base_path = (
    r"C:\Users\Mille\.cache\huggingface\hub"
    r"\datasets--PulpBio--SilentWear"
    r"\snapshots\cb1f0d1fb2688c0cd0a3f21336a04e86bf906096"
    r"\data_raw_and_filt"
)

SUBJECTS = ["S01", "S02", "S03", "S04"]
MODES    = ["silent", "vocalized"]

#falam start antes de falar os comandos(protocol cue)  -> fazer experimentos com e sem ela 
PROTOCOL_CUE = "start"

#extração de features
def extract_features(signal: np.ndarray) -> np.ndarray:
    THRESHOLD_ZC = 0.01
    THRESHOLD_SSC = 0.01
    THRESHOLD_WAMP = 0.01

    d = np.diff(signal, axis=0)
    abs_d = np.abs(d)

    rms = np.sqrt(np.mean(signal**2, axis=0))
    mav = np.mean(np.abs(signal), axis=0)
    std = np.std(signal, axis=0, ddof=1)
    wl = np.sum(abs_d, axis=0)

    d_sign = np.diff(np.sign(signal), axis=0)
    zc = np.sum((d_sign != 0) & (abs_d >= THRESHOLD_ZC), axis=0).astype(float)

    d_sign2 = np.diff(np.sign(d), axis=0)
    ssc = np.sum((d_sign2 != 0) & ((np.abs(d[:-1]) >= THRESHOLD_SSC) | (np.abs(d[1:]) >= THRESHOLD_SSC)), axis=0).astype(float)

    iav = np.sum(np.abs(signal), axis=0)
    var = np.var(signal, axis=0, ddof=1)
    dasdv = np.std(d, axis=0, ddof=1)
    msr = np.mean(np.sqrt(np.abs(signal)), axis=0)

    sum_sq_diff = np.sum(d **2, axis=0)
    mfl = np.log10(np.sqrt(np.maximum(sum_sq_diff, 1e-10)))
    wamp = np.sum(abs_d >= THRESHOLD_WAMP, axis=0).astype(float)

    n = signal.shape[0]
    x_sorted = np.sort(signal, axis=0)
    weights = (np.arange(n) / (n - 1))[:, np.newaxis]
    b0 = x_sorted.mean(axis=0)
    b1 = (weights * x_sorted).mean(axis=0)
    ls = 2*b1-b0

    return np.concatenate([rms, mav, std, wl, zc, ssc, iav, var, dasdv, msr, mfl, wamp, ls])

#carregando e segmentando
def carregamento(base_path: str, subjects: list, mode: str) -> pd.DataFrame:
    records = []

    for subject in subjects:
        folder = os.path.join(base_path, subject, mode)
        files = sorted(f for f in os.listdir(folder) if f.endswith(".h5"))

        print(f"carregando {subject}/{mode} --> {len(files)}")

        for file in files:
            path = os.path.join(folder, file)
            df = pd.read_hdf(path)

            #colunas EMG filtradas
            filt_cols = [c for c in df.columns if "filt" in c.lower()]
            if "Label_str" not in df.columns:
                print(f"Sem Label_str: {file} ignorado.")
                continue

            labels = df["Label_str"].values
            signal_all = df[filt_cols].values

            #detecção de limites de eventos via mudança de label
            event_ids = np.concatenate([[0], np.cumsum(labels[1:] != labels[:-1])])
            change_pts = np.where(np.diff(event_ids))[0] + 1
            starts = np.concatenate([[0], change_pts])
            ends = np.concatenate([change_pts, [len(labels)]])

            for ev_id, (s, e) in enumerate(zip(starts, ends)):
                n_samples = e - s
                word = labels[s]

                #descarta transições muito curtas
                if n_samples < 200:
                    continue

                segment = signal_all[s:e]

                #normalização z-score por canal dentro do evento
                mean = segment.mean(axis=0)
                std = segment.std(axis=0)
                std[std == 0] = 1
                segment_norm = (segment - mean) / std

                feat = extract_features(segment_norm)

                records.append({"subject":  subject, "mode": mode, "trial": file, "event_id": ev_id, "word": word, "n_samples": n_samples, **{f"feat_{i}": v for i, v in enumerate(feat)},})

    return pd.DataFrame(records)

#remoção de outliers
def remove_outliers(df: pd.DataFrame, feat_cols: list, pct: float = 99) -> pd.DataFrame:
    X = df[feat_cols].values
    max_feat = np.max(np.abs(X), axis=1)
    threshold = np.percentile(max_feat, pct)
    mask = max_feat < threshold
    return df[mask].reset_index(drop=True)

#acuracia e f1-score
def compute_metrics(y_true, y_pred, labels=None) -> dict:
    return {"accuracy":  accuracy_score(y_true, y_pred), "macro_f1":  f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),}


#classificadores 
#nao usamos hiperparametros pois o objetivo eh comparar eles com os settings no default
def build_classifier(model_name: str):
    if model_name == "LDA":
        return LinearDiscriminantAnalysis(solver="svd")

    elif model_name == "SVM":
        return SVC(
            kernel="rbf",
            C=1,
            gamma="scale"
        )

    elif model_name == "RF":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Classificador desconhecido: {model_name}")

def print_fold_header(fold_name: str):
    print(f"Fold: {fold_name}")

def print_metrics(metrics: dict, n_train: int, n_test: int):
    print(f"Treino: {n_train} eventos --- Teste: {n_test} eventos")
    print(f"Acurácia: {metrics['accuracy']*100:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

#loso --> 4 individuos. 3 sao treino e o ultimo eh teste
def run_loso(df: pd.DataFrame, feat_cols: list, classifier_name: str, label_col: str = "word") -> list:
    results = []

    for test_subj in sorted(df["subject"].unique()):
        train_mask = df["subject"] != test_subj
        test_mask = df["subject"] == test_subj

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, label_col].values
        X_test = df.loc[test_mask,  feat_cols].values
        y_test = df.loc[test_mask,  label_col].values

        #scaler fitado APENAS no treino
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = build_classifier(classifier_name)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        print_fold_header(f"Teste = {test_subj}")
        print_metrics(metrics, len(y_train), len(y_test))

        #relatório por classe
        print("\n" + classification_report(y_test, y_pred, zero_division=0))

        results.append({"fold": test_subj, "n_train": len(y_train), "n_test": len(y_test), **metrics, })
    return results

#loto ---> cada um dos 4 individuos tem um fold de teste e outro de treino
def run_loto(df: pd.DataFrame, feat_cols: list, classifier_name: str, label_col: str = "word") -> list:
    results = []

    for subject in sorted(df["subject"].unique()):
        subj_df = df[df["subject"] == subject].reset_index(drop=True)
        trials = sorted(subj_df["trial"].unique())
        print(f"\n  Sujeito {subject} : {len(trials)} trials")

        for test_trial in trials:
            train_mask = subj_df["trial"] != test_trial
            test_mask = subj_df["trial"] == test_trial

            X_train = subj_df.loc[train_mask, feat_cols].values
            y_train = subj_df.loc[train_mask, label_col].values
            X_test = subj_df.loc[test_mask,  feat_cols].values
            y_test = subj_df.loc[test_mask,  label_col].values

            #garante que todas as classes do teste estejam no treino
            classes_train = set(np.unique(y_train))
            classes_test = set(np.unique(y_test))
            missing = classes_test - classes_train
            if missing:
                print(f"{test_trial}: classes {missing} ausentes no treino, fold ignorado.")
                continue

            #scaler fitado APENAS no treino
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = build_classifier(classifier_name)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            metrics = compute_metrics(y_test, y_pred)
            print_fold_header(f"Teste = {test_trial}")
            print_metrics(metrics, len(y_train), len(y_test))

            results.append({"subject": subject, "fold": test_trial, "n_train": len(y_train), "n_test": len(y_test), **metrics,})

    return results

#imprime as metricas (acurácia e macro f1) 
def summarize(results: list, label: str):
    df_r = pd.DataFrame(results)
    print(f"\nResumo {label}")
    print(f"Acurácia: {df_r['accuracy'].mean()*100:.2f}% +- {df_r['accuracy'].std()*100:.2f}%")
    print(f"Macro-F1: {df_r['macro_f1'].mean():.4f} +- {df_r['macro_f1'].std():.4f}")
    return df_r

def define_experiments(include_start: bool) -> list:
    cue = [PROTOCOL_CUE] if not include_start else []
    nav_commands = ["up", "down", "left", "right", "stop", "forward", "backward"]
    if include_start:
        nav_commands.append("start")

    experiments = [
        {"name": "EXP1_fala_vs_repouso", "description": "Fala (todos os comandos) vs. Repouso", "exclude": cue, "binary_rest": True,},
        {"name": "EXP2_entre_frases", "description": "Entre frases (sem repouso)", "exclude": cue + ["rest"], "binary_rest": False,},
        {"name": "EXP3_tudo", "description": "Tudo (comandos + repouso)", "exclude": cue, "binary_rest": False,},]
    return experiments


def main():
    all_results = []  # acumula p exportação
    for mode in MODES:
        print(f"MODO: {mode.upper()}")

        #carrega e extrai features uma única vez por modo
        df_mode = carregamento(base_path, SUBJECTS, mode)
        feat_cols = [c for c in df_mode.columns if c.startswith("feat_")]

        #remove outliers globais (percentil 99 do max abs de cada evento)
        df_mode = remove_outliers(df_mode, feat_cols)
        print(f"\nTotal de eventos após remoção de outliers: {len(df_mode)}")
        print("\nDistribuição de classes:")
        print(df_mode["word"].value_counts().to_string())

        #com e sem start (protocol cue dos caras do dataset)
        for include_start in [False, True]:
            scenario_label = "COM start" if include_start else "SEM start"
            experiments = define_experiments(include_start)
            for exp in experiments:
                print(f"{exp['name']} — {exp['description']}")
                print(f"[{scenario_label}]")

                #filtragem das classes do experimento
                df_exp = df_mode[~df_mode["word"].isin(exp["exclude"])].copy()

                #fala vs descanso 
                if exp["binary_rest"]:
                    df_exp["word"] = df_exp["word"].apply(
                        lambda w: "rest" if w == "rest" else "speech"
                    )

                classes_present = sorted(df_exp["word"].unique())
                print(f"\nClasses: {classes_present}")
                print(f"Eventos: {len(df_exp)}")

                #LDA e SVM
                for classifier_name in ["LDA", "SVM", "RF"]:
                    print(f"\nCLASSIFICADOR: {classifier_name}")

                    # LOSO
                    print(f"\nLOSO (inter-indivíduo)")
                    loso_results = run_loso(df_exp, feat_cols, classifier_name)

                    df_loso = summarize(loso_results, f"LOSO: {classifier_name}")

                    # LOTO
                    print(f"\nLOTO (intra-indivíduo)")
                    loto_results = run_loto(df_exp, feat_cols, classifier_name)

                    df_loto = summarize(loto_results, f"LOTO: {classifier_name}")

                    for r in loso_results:
                        r.update({"classifier": classifier_name, "mode": mode, "scenario": scenario_label, "experiment": exp["name"], "cv": "LOSO"})

                    for r in loto_results:
                        r.update({"classifier": classifier_name, "mode": mode, "scenario": scenario_label, "experiment": exp["name"], "cv": "LOTO"})

                    all_results.extend(loso_results)
                    all_results.extend(loto_results)

    #tabela final
    df_all = pd.DataFrame(all_results)
    summary = (df_all.groupby(["classifier", "mode", "scenario", "experiment", "cv"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4))
    print("\nRESUMO GERAL")
    print(summary.to_string())

    #salva na mesma pasta do script pq tava dando permissionerror KKKKKKK
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifiers_results_all_folds.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\nResultados completos salvos em: {out_csv}")


if __name__ == "__main__":
    main()