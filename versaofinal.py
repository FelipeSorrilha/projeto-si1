import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier


#caminho raiz pro dataset (pra rodar tem que trocar)
base_path = (
    r"C:\Users\Mille\.cache\huggingface\hub"
    r"\datasets--PulpBio--SilentWear"
    r"\snapshots\cb1f0d1fb2688c0cd0a3f21336a04e86bf906096"
    r"\data_raw_and_filt"
)

#sujeitos do dataset
SUBJECTS = ["S01", "S02", "S03", "S04"]

MODES = ["silent", "vocalized"]

#o pessoal fala start antes de falar o comando. a gente vai rodar experimentos com e sem ele 
PROTOCOL_CUE = "start"

#extração de features (vetor de 208 dimensoes ja que 16 canais x 13 features)
def extract_features(signal: np.ndarray) -> np.ndarray:
    #limiares usados por features baseadas em cruzamento de zero/limiar
    THRESHOLD_ZC   = 0.01 
    THRESHOLD_SSC  = 0.01 
    THRESHOLD_WAMP = 0.01 

    #diferença de primeira ordem (entre amostras consecutivas)
    d = np.diff(signal, axis=0)
    abs_d = np.abs(d)


    #RMS:potência média do sinal
    rms = np.sqrt(np.mean(signal**2, axis=0))

    # MAV:valor absoluto médio
    mav = np.mean(np.abs(signal), axis=0)

    #STD:desvio padrão do sinal
    std = np.std(signal, axis=0, ddof=1)

    #WL:comprimento total da forma de onda (soma das variações)
    wl = np.sum(abs_d, axis=0)

    #ZC:número de vezes que o sinal cruza o zero (acima do limiar)
    d_sign = np.diff(np.sign(signal), axis=0)
    zc = np.sum((d_sign != 0) & (abs_d >= THRESHOLD_ZC), axis=0).astype(float)

    #SSC:mudanças no sentido da inclinação (picos e vales)
    d_sign2 = np.diff(np.sign(d), axis=0)
    ssc = np.sum((d_sign2 != 0) & ((np.abs(d[:-1]) >= THRESHOLD_SSC) | (np.abs(d[1:]) >= THRESHOLD_SSC)), axis=0).astype(float)

    #IAV:soma dos valores absolutos
    iav = np.sum(np.abs(signal), axis=0)

    #VAR:variância do sinal
    var = np.var(signal, axis=0, ddof=1)

    #DASDV:diferença de desvio padrão entre amostras consecutivas
    dasdv = np.std(d, axis=0, ddof=1)

    #MSR:média das raízes quadradas dos valores absolutos
    msr = np.mean(np.sqrt(np.abs(signal)), axis=0)

    #MFL:estimativa da dimensão fractal pelo log da energia da diferença
    sum_sq_diff = np.sum(d**2, axis=0)
    mfl = np.log10(np.sqrt(np.maximum(sum_sq_diff, 1e-10)))  # clamp evita log(0)

    #WAMP:conta quantas variações entre amostras superam o limiar
    wamp = np.sum(abs_d >= THRESHOLD_WAMP, axis=0).astype(float)

    #LS:estimativa robusta de localização
    n = signal.shape[0]
    x_sorted = np.sort(signal, axis=0)
    weights = (np.arange(n) / (n - 1))[:, np.newaxis]  #pesos lineares em [0, 1]
    b0 = x_sorted.mean(axis=0)
    b1 = (weights * x_sorted).mean(axis=0)
    ls = 2*b1-b0  #combinação linear de estimadores de posição

    #concatena todas as 13 features para os n_canais num vetor de 208 dims (13 × 16)
    return np.concatenate([rms, mav, std, wl, zc, ssc, iav, var, dasdv, msr, mfl, wamp, ls])

#carregando o dataset --> segmenta os eventos por label e extrai as features de cada segmento dai
def carregamento(base_path, subjects, mode):
    records = []

    for subject in subjects:
        folder = os.path.join(base_path, subject, mode)
        #lista apenas os arquivos .h5
        files = sorted(f for f in os.listdir(folder) if f.endswith(".h5"))
        print(f"carregando {subject}/{mode} --> {len(files)}")

        for file in files:
            path = os.path.join(folder, file)
            df = pd.read_hdf(path)

            #seleciona apenas as colunas de sinal filtrado --> eles tem filt no nome
            filt_cols = [c for c in df.columns if "filt" in c.lower()]
            if "Label_str" not in df.columns:
                print(f"Sem Label_str: {file} ignorado.")
                continue

            labels = df["Label_str"].values
            signal_all = df[filt_cols].values

            #segmentação por eventos
            #add um id crescente a cada bloco contiguo c o mesmo label
            event_ids = np.concatenate([[0], np.cumsum(labels[1:] != labels[:-1])])
            #encontra os indices onde o label muda --> evento novo
            change_pts = np.where(np.diff(event_ids))[0] + 1
            starts = np.concatenate([[0], change_pts])
            ends = np.concatenate([change_pts, [len(labels)]])

            for ev_id, (s, e) in enumerate(zip(starts, ends)):
                n_samples = e-s
                word = labels[s]  #label do evento

                #descarta segmentos muito curtos
                if n_samples < 200:
                    continue

                #normalização dentro do segmento
                segment = signal_all[s:e]
                mean = segment.mean(axis=0)
                std_seg = segment.std(axis=0)
                std_seg[std_seg == 0] = 1        #evita divisão por zero em canais planos
                segment_norm = (segment - mean) / std_seg

                #extrai o vetor de features e registra o evento
                feat = extract_features(segment_norm)
                records.append({"subject": subject, "mode": mode, "trial": file, "event_id": ev_id, "word": word, "n_samples": n_samples, **{f"feat_{i}": v for i, v in enumerate(feat)},})

    return pd.DataFrame(records)

#remocao de outliers. amostras com valor max absoluto de feature acima do percentil sao removidas
def remove_outliers(df, feat_cols, pct=99):
    X = df[feat_cols].values
    max_feat = np.max(np.abs(X), axis=1)         #maior valor absoluto por amostra
    threshold = np.percentile(max_feat, pct)       #limiar no percentil desejado
    return df[max_feat < threshold].reset_index(drop=True)

#metricas --> acuracia e macro f1
def compute_metrics(y_true, y_pred, labels=None):
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),}

#pra imprimir o cabecalho de cada fold
def print_fold_header(fold_name):
    print(f"Fold: {fold_name}")

def print_metrics(metrics, n_train, n_test):
    print(f"Treino: {n_train} eventos --- Teste: {n_test} eventos")
    print(f"Acurácia: {metrics['accuracy']*100:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")


#classificadores
def classif(model_name):
    if model_name == "LDA":
        return LinearDiscriminantAnalysis(solver="svd")
    elif model_name == "SVM":
        return SVC(kernel="rbf", C=1, gamma="scale")
    elif model_name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Classificador desconhecido: {model_name}")

#loso --> 3 sujeitos sao treino e o ultimo eh teste 
def loso(df, feat_cols, classifier_name, label_col="word"):
    results = []
    all_y_true, all_y_pred = [], []

    for test_subj in sorted(df["subject"].unique()):
        #separa treino (todos menos o sujeito de teste) e teste
        train_mask = df["subject"] != test_subj
        test_mask = df["subject"] == test_subj

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, label_col].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, label_col].values

        #normalização
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #treinamento e predicao
        clf = classif(classifier_name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        metrics = compute_metrics(y_test, y_pred)
        print_fold_header(f"Teste = {test_subj}")
        print_metrics(metrics, len(y_train), len(y_test))
        print("\n" + classification_report(y_test, y_pred, zero_division=0))

        results.append({"fold": test_subj, "n_train": len(y_train), "n_test": len(y_test), **metrics})

    return results, np.array(all_y_true), np.array(all_y_pred)

#loto --> cada sujeito tem um fold com treino e com teste
def run_loto(df, feat_cols, classifier_name, label_col="word"):
    results = []
    all_y_true, all_y_pred = [], []

    for subject in sorted(df["subject"].unique()):
        subj_df = df[df["subject"] == subject].reset_index(drop=True)
        trials = sorted(subj_df["trial"].unique())
        print(f"\nSujeito {subject}: {len(trials)} trials")

        for test_trial in trials:
            train_mask = subj_df["trial"] != test_trial
            test_mask = subj_df["trial"] == test_trial

            X_train = subj_df.loc[train_mask, feat_cols].values
            y_train = subj_df.loc[train_mask, label_col].values
            X_test = subj_df.loc[test_mask, feat_cols].values
            y_test = subj_df.loc[test_mask, label_col].values

            # Fold ignorado se o conjunto de teste contém classes ausentes no treino
            # (o classificador não conseguiria prever tais classes corretamente)
            missing = set(np.unique(y_test)) - set(np.unique(y_train))
            if missing:
                print(f"{test_trial}: classes {missing} ausentes no treino, fold ignorado.")
                continue

            # Normalização intra-sujeito (sem leakage)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = classif(classifier_name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            metrics = compute_metrics(y_test, y_pred)
            print_fold_header(f"Teste = {test_trial}")
            print_metrics(metrics, len(y_train), len(y_test))

            results.append({"subject": subject, "fold": test_trial, "n_train": len(y_train), "n_test":  len(y_test), **metrics})
    return results, np.array(all_y_true), np.array(all_y_pred)

#resumao dos resultados --> mostra media e desvio padrao pra cada fold
def summarize(results, label):
    df_r = pd.DataFrame(results)
    print(f"\nResumo {label}")
    print(f"Acurácia: {df_r['accuracy'].mean()*100:.2f}% +- {df_r['accuracy'].std()*100:.2f}%")
    print(f"Macro-F1: {df_r['macro_f1'].mean():.4f} +- {df_r['macro_f1'].std():.4f}")
    return df_r

#definindo os 3 exps --> fala vs rest, so fala e tudo junto
def define_experiments(include_start):
    #se não deve incluir o "start", adiciona a lista de exclusão de cada experimento
    cue = [PROTOCOL_CUE] if not include_start else []

    experiments = [{"name": "EXP1_fala_vs_repouso", "description": "Fala vs. Repouso", "exclude": cue, "binary_rest": True},
        {"name": "EXP2_entre_frases", "description": "Entre frases (sem repouso)", "exclude": cue + ["rest"], "binary_rest": False},
        {"name": "EXP3_tudo", "description": "Tudo (comandos + repouso)", "exclude": cue, "binary_rest": False},]
    return experiments

#matrizes de confusao ---> a coluna sao os exps e as linhas sao silent/sem silent/com, vocalized/sem vocalized/com
def plot_confusion_matrices(cm_store, classifier_name, cv, out_dir):
    #define a ordem das linhas e seus rotulos
    row_keys = [("silent", "SEM start"), ("silent", "COM start"), ("vocalized", "SEM start"), ("vocalized", "COM start"),]
    row_labels = ["silent / SEM start", "silent / COM start", "vocal / SEM start",  "vocal / COM start"]

    exp_names = ["EXP1_fala_vs_repouso", "EXP2_entre_frases", "EXP3_tudo"]
    exp_labels = ["EXP1\nFala vs Repouso", "EXP2\nEntre Frases", "EXP3\nTudo"]

    n_rows, n_cols = len(row_keys), len(exp_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))

    for row_i, (mode, scenario) in enumerate(row_keys):
        entries = cm_store.get((mode, scenario), [])

        for col_i, exp_name in enumerate(exp_names):
            ax = axes[row_i, col_i]
            entry = next((e for e in entries if e["exp_name"] == exp_name), None)

            #titulo de coluna apenas na primeira linha
            if row_i == 0:
                ax.set_title(exp_labels[col_i], fontsize=10, fontweight="bold", pad=8)

            #rotulo de linha apenas na primeira coluna
            if col_i == 0:
                ax.set_ylabel(row_labels[row_i], fontsize=9, labelpad=8)

            #eixo vazio se n tem dados para essa combinação
            if entry is None or len(entry["y_true"]) == 0:
                ax.axis("off")
                continue

            #calcula e normaliza a matriz de confusão
            classes = sorted(set(entry["y_true"]) | set(entry["y_pred"]))
            cm = confusion_matrix(entry["y_true"], entry["y_pred"], labels=classes)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)
            disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel("")   # remove xlabel padrão do sklearn

    fig.suptitle(f"Matrizes de Confusão: {classifier_name} - {cv}", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"cm_{classifier_name}_{cv}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out_path}")

#grafico de evolução de complexidade
def plot_evolution(df_all, out_dir):
    CLASSIFIERS = ["LDA", "SVM", "RF"]
    SCENARIOS = ["SEM start", "COM start"]
    EXP_ORDER = ["EXP1_fala_vs_repouso", "EXP2_entre_frases", "EXP3_tudo"]
    EXP_LABELS = ["EXP1\nFala vs Repouso", "EXP2\nEntre Frases", "EXP3\nTudo"]
    COLORS = {"LDA": "#4878CF", "SVM": "#E87C25", "RF": "#56A55A"}
    MARKERS = {"LDA": "o", "SVM": "s", "RF": "^"}

    #filtra apenas os resultados de vocalized + LOTO
    df_plot = df_all[(df_all["mode"] == "vocalized") & (df_all["cv"] == "LOTO")].copy()

    #agrega média e desvio padrão de Macro-F1 por (classificador, cenário, experimento)
    df_agg = (df_plot.groupby(["classifier", "scenario", "experiment"])["macro_f1"].agg(["mean", "std"]).reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Progressão de Complexidade: Macro-F1\nVocalized | LOTO", fontsize=13, fontweight="bold")

    for col_i, scenario in enumerate(SCENARIOS):
        ax = axes[col_i]
        df_scene = df_agg[df_agg["scenario"] == scenario]

        for clf in CLASSIFIERS:
            df_clf = df_scene[df_scene["classifier"] == clf]
            df_clf = df_clf.set_index("experiment").reindex(EXP_ORDER).reset_index()

            means = df_clf["mean"].values
            stds = df_clf["std"].values
            x = np.arange(len(EXP_ORDER))

            ax.plot(x, means, color=COLORS[clf], marker=MARKERS[clf], linewidth=2, markersize=8, label=clf)

            #banda de incerteza (+-1 std)
            ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=COLORS[clf])

        ax.set_title(scenario, fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(EXP_ORDER)))
        ax.set_xticklabels(EXP_LABELS, fontsize=9)
        ax.set_ylabel("Macro-F1" if col_i == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="x", linestyle=":",  alpha=0.3)
        ax.legend(title="Classificador", fontsize=9)

        #anota o valor da media sobre cada ponto do grafico
        for clf in CLASSIFIERS:
            df_clf = df_scene[df_scene["classifier"] == clf]
            df_clf = df_clf.set_index("experiment").reindex(EXP_ORDER).reset_index()
            for xi, (mean, std) in enumerate(zip(df_clf["mean"], df_clf["std"])):
                ax.annotate(f"{mean:.2f}", xy=(xi, mean), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7.5, color=COLORS[clf])

    plt.tight_layout()
    out_path = os.path.join(out_dir, "evolution_vocalized_loto.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out_path}")

#main
def main():
    all_results = []
    out_dir = os.path.dirname(os.path.abspath(__file__))

    cm_store = {clf: {"LOSO": {}, "LOTO": {}} for clf in ["LDA", "SVM", "RF"]}

    for mode in MODES:
        #carrega e pre-processa os dados do modo atual
        df_mode = carregamento(base_path, SUBJECTS, mode)
        feat_cols = [c for c in df_mode.columns if c.startswith("feat_")]
        df_mode = remove_outliers(df_mode, feat_cols)

        print(f"\nTotal após remoção de outliers: {len(df_mode)}")
        print(df_mode["word"].value_counts().to_string())

        #itera pelos dois cenários: sem e com o sinal START
        for include_start in [False, True]:
            scenario_label = "COM start" if include_start else "SEM start"
            experiments = define_experiments(include_start)

            for classifier_name in ["LDA", "SVM", "RF"]:
                print(f"CLASSIFICADOR: {classifier_name}, {mode}, {scenario_label}")

                for exp in experiments:
                    print(f"\n{exp['name']} — {exp['description']}")

                    #filtra classes excluídas do experimento
                    df_exp = df_mode[~df_mode["word"].isin(exp["exclude"])].copy()

                    #EXP1
                    if exp["binary_rest"]:
                        df_exp["word"] = df_exp["word"].apply(lambda w: "rest" if w == "rest" else "speech")

                    classes_present = sorted(df_exp["word"].unique())
                    print(f"Classes: {classes_present}, Eventos: {len(df_exp)}")

                    #LOSO
                    print(f"\nLOSO")
                    loso_results, loso_yt, loso_yp = loso(df_exp, feat_cols, classifier_name)
                    summarize(loso_results, f"LOSO {classifier_name}")

                    #LOTO
                    print(f"\nLOTO")
                    loto_results, loto_yt, loto_yp = run_loto(df_exp, feat_cols, classifier_name)
                    summarize(loto_results, f"LOTO {classifier_name}")

                    #add metadados e acumula resultados para o CSV final
                    for r in loso_results:
                        r.update({"classifier": classifier_name, "mode": mode, "scenario": scenario_label, "experiment": exp["name"],"cv": "LOSO"})
                    for r in loto_results:
                        r.update({"classifier": classifier_name, "mode": mode, "scenario": scenario_label, "experiment": exp["name"],"cv": "LOTO"})
                    all_results.extend(loso_results)
                    all_results.extend(loto_results)

                    #armazena predicoes para gerar as matrizes de confusão
                    key = (mode, scenario_label)
                    for cv_label, yt, yp in [("LOSO", loso_yt, loso_yp), ("LOTO", loto_yt, loto_yp)]:
                        store = cm_store[classifier_name][cv_label]
                        if key not in store:
                            store[key] = []
                        store[key].append({"exp_name": exp["name"], "y_true": yt, "y_pred": yp,})

    #geracao das matrizes de confusao
    for classifier_name in ["LDA", "SVM", "RF"]:
        for cv in ["LOSO", "LOTO"]:
            plot_confusion_matrices(cm_store[classifier_name][cv], classifier_name, cv, out_dir)

    #csv com os dados brutos
    df_all  = pd.DataFrame(all_results)
    summary = (df_all.groupby(["classifier", "mode", "scenario", "experiment", "cv"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4))
    print("\nRESUMO GERAL")
    print(summary.to_string())

    out_csv = os.path.join(out_dir, "classifiers_results_all_folds.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\nResultados salvos em: {out_csv}")

    # grafico de evolucao de complex
    plot_evolution(df_all, out_dir)


if __name__ == "__main__":
    main()