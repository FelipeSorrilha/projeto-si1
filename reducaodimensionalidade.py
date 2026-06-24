import os
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

#pra rodar tem que colocar o caminho do arquivo aqui
base_path = (
    r"C:\Users\Mille\.cache\huggingface\hub"
    r"\datasets--PulpBio--SilentWear"
    r"\snapshots\cb1f0d1fb2688c0cd0a3f21336a04e86bf906096"
    r"\data_raw_and_filt"
)

#numero max pq senao demorava MTO pra rodar
MAX_UMAP_PTS = 10000
MAX_TSNE_PTS = 5000

subjects = ["S01", "S02", "S03", "S04"]
modes    = ["silent", "vocalized"]

#isso aqui eh pro grafico que ele gera pra pca, umap e t-sne
WORD_COLORS = {
    "up":       "#e6194b",
    "down":     "#3cb44b",
    "left":     "#4363d8",
    "right":    "#f58231",
    "start":    "#911eb4",
    "stop":     "#42d4f4",
    "forward":  "#f032e6",
    "backward": "#bfef45",
    "rest":     "#808080",
}
DEFAULT_COLOR = "#aaaaaa"

#extração de features (rms, mav, std, wl, zc, ssc, iav, var, dasdv, msr, mfl, wamp, ls) por canal
#como sao 13 features e 16 canais, 16x13 = 208
def extract_features(signal: np.ndarray) -> np.ndarray:
    #limiares pras features que precisam
    THRESHOLD_ZC   = 0.01
    THRESHOLD_SSC  = 0.01
    THRESHOLD_WAMP = 0.01  

    d = np.diff(signal, axis=0)    #primeira diferença (n-1, n_can)
    d2 = np.diff(d, axis=0)    #segunda diferença (n-2, n_can)
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

    sum_sq_diff = np.sum(d**2, axis=0)
    mfl = np.log10(np.sqrt(np.maximum(sum_sq_diff, 1e-10)))
    wamp = np.sum(abs_d >= THRESHOLD_WAMP, axis=0).astype(float)

    n = signal.shape[0]
    x_sorted = np.sort(signal, axis=0)                      
    weights = (np.arange(n) / (n-1))[:, np.newaxis]   
    b0 = x_sorted.mean(axis=0)
    b1 = (weights * x_sorted).mean(axis=0)
    ls = 2*b1-b0

    #concatenação das features
    return np.concatenate([rms, mav, std, wl, zc, ssc, iav, var, dasdv, msr, mfl, wamp, ls])


#processamento
def process_file(df: pd.DataFrame, filt_cols: list, subject: str, file: str) -> tuple:
    feats = []
    metas = []

    #identifica limites dos eventos via diferença de label
    labels = df["Label_str"].values
    event_ids = np.concatenate([[0], np.cumsum(labels[1:] != labels[:-1])])

    #indices onde cada evento começa e termina
    change_pts = np.where(np.diff(event_ids))[0] + 1
    starts = np.concatenate([[0], change_pts])
    ends = np.concatenate([change_pts, [len(labels)]])

    #aqui o vetor completo do arquivo
    signal_all = df[filt_cols].values   

    for ev_id, (s, e) in enumerate(zip(starts, ends)):
        n_samples = e - s
        word = labels[s]

        #descarta transições muito curtas (<100ms = 200 amostras)
        if n_samples < 200:
            continue

        segment = signal_all[s:e]  

        #normalização por canal dentro do evento
        mean = segment.mean(axis=0)
        std = segment.std(axis=0)
        std[std == 0] = 1
        signal_norm = (segment - mean) / std

        #pega as features do sinal normalizado
        feat = extract_features(signal_norm)

        feats.append(feat)
        metas.append({"subject": subject, "word": word, "file": file, "event_id": ev_id, "n_samples": n_samples,})
    return feats, metas

#subsample estratificado
def stratified_sample(X, meta, n_max, seed=42):
    rng = np.random.default_rng(seed)
    n_tot = len(X)

    if n_tot <= n_max:
        return X, meta, np.arange(n_tot)

    idx_sub = []
    for word in meta["word"].unique():
        idx_word = np.where(meta["word"] == word)[0]
        n_word = max(1, round(n_max * len(idx_word) / n_tot))
        chosen = rng.choice(idx_word, min(n_word, len(idx_word)), replace=False)
        idx_sub.extend(chosen.tolist())

    idx_sub = np.array(idx_sub)
    return X[idx_sub], meta.iloc[idx_sub].reset_index(drop=True), idx_sub

#estrutura de saída
all_features = {m: [] for m in modes}
all_meta     = {m: [] for m in modes}

#carregamento e segmentação por evento
for subject in subjects:
    for mode in modes:
        folder = os.path.join(base_path, subject, mode)
        files = sorted(f for f in os.listdir(folder) if f.endswith(".h5"))

        #pra saber se ta carregando os arquivos kkkk
        print(f"{subject}/{mode}: {len(files)} arquivo(s)")

        for i, file in enumerate(files, 1):
            path = os.path.join(folder, file)
            df = pd.read_hdf(path)

            filt_cols = [c for c in df.columns if "filt" in c.lower()]

            if "Label_str" not in df.columns:
                print(f"Sem Label_str:{file} foi ignorado.")
                continue

            feats, metas = process_file(df, filt_cols, subject, file)

            all_features[mode].extend(feats)
            all_meta[mode].extend(metas)
            print(f"[{i}/{len(files)}] {file} --> {len(feats)} eventos", flush=True)

print("\nEventos extraídos por modo:")
for mode in modes:
    print(f"{mode}: {len(all_features[mode])} eventos")

#pipeline por classe
for mode in modes:
    print(f"MODO: {mode}")

    X = np.array(all_features[mode])
    meta = pd.DataFrame(all_meta[mode])

    #remoção de outliers
    max_feat = np.max(np.abs(X), axis=1)
    threshold = np.percentile(max_feat, 99)
    mask = max_feat < threshold

    X = X[mask]
    meta = meta[mask].reset_index(drop=True)

    print(f"Eventos após remoção de outliers: {X.shape[0]}")
    print(f"Dimensão do vetor de features: {X.shape[1]} " f"({X.shape[1] // 13} canais x 13 features: " f"RMS, MAV, STD, WL, ZC, SSC, IAV, VAR, DASDV, MSR, MFL, WAMP, LS)")

    #normalização global
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #pca
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Variância explicada PCA: {pca.explained_variance_ratio_.round(4)}")

    #subsample compartilhado
    n_shared = min(MAX_UMAP_PTS, MAX_TSNE_PTS)
    X_sub, meta_sub, idx_sub = stratified_sample(X_scaled, meta, n_shared)
    X_pca_sub = X_pca[idx_sub]
    print(f"Pontos usados em UMAP e t-SNE: {len(X_sub)}")

    #umap
    X_umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", low_memory=True, n_jobs=-1,).fit_transform(X_sub)
    print("UMAP concluído")

    #t-sne
    sk_version  = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    tsne_kwargs = dict(n_components=2, perplexity=30, init="pca", method="barnes_hut", random_state=42, n_jobs=-1,)
    tsne_kwargs["max_iter" if sk_version >= (1, 4) else "n_iter"] = 500
    X_tsne = TSNE(**tsne_kwargs).fit_transform(X_sub)
    print("t-SNE concluído.")

    #plotando o pca, umap e t-sne
    words_present = sorted(meta_sub["word"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"Features por Evento: {mode}" f"(RMS, MAV, STD, WL, ZC, SSC, IAV, VAR, DASDV, MSR, MFL, WAMP, LS {len(X_sub)} eventos)", fontsize=12, fontweight="bold")

    plot_data = [(axes[0], X_pca_sub, "PCA", f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"), (axes[1], X_umap, "UMAP", "Dimensão 1", "Dimensão 2"), (axes[2], X_tsne, "t-SNE", "Dimensão 1", "Dimensão 2"),]

    for ax, X_proj, title, xlabel, ylabel in plot_data:
        for word in words_present:
            mask_w = meta_sub["word"] == word
            color = WORD_COLORS.get(word, DEFAULT_COLOR)
            ax.scatter(X_proj[mask_w, 0], X_proj[mask_w, 1], label=word, color=color, alpha=0.7, s=30, linewidths=0,)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(markerscale=2, fontsize=8, title="Palavra")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"pca_umap_tsne_{mode}.png", dpi=150, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.5)

    print(f"Plot salvo: pca_umap_tsne_{mode}.png")

#enter pra fechar as janelas
input("\nPressione Enter p encerrar")
