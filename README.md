# Silent & Vocalized Speech Classification via Surface EMG

Classificação de comandos de fala silenciosa e vocalizada usando sinais de eletromiografia de superfície (sEMG) e algoritmos de Machine Learning (LDA, SVM e Random Forest).

> Projeto desenvolvido para a disciplina de Sistemas Inteligentes — Engenharia da Computação, UTFPR Apucarana.

# Alunos:
> Felipe Ferrer Sorrilha

> João Antônio Sitta Martins

> Millena Sartori de Oliveira

---

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Dataset](#dataset)
- [Instalação](#instalação)
- [Como Rodar](#como-rodar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pipeline](#pipeline)
- [Resultados](#resultados)

---

## Sobre o Projeto

Este projeto investiga a viabilidade de reconhecer comandos de fala diretamente a partir de sinais musculares, sem depender de áudio. A motivação vem de aplicações assistivas para pessoas com dificuldades de comunicação — como após laringectomias, traqueostomias ou distúrbios neurológicos.

A abordagem extrai **13 features no domínio do tempo** (RMS, MAV, STD, WL, ZC, SSC, IAV, VAR, DASDV, MSR, MFL, WAMP, LS) de **16 canais de EMG**, gerando vetores de 208 dimensões por evento. Três classificadores clássicos são comparados (LDA, SVM, RF) em múltiplos cenários, com validação LOSO e LOTO.

---

## Dataset

O projeto usa o **SilentWear**, um dataset público de EMG multi-sessão para reconhecimento de fala silenciosa e vocalizada.

- **Link:** https://huggingface.co/datasets/PulpBio/SilentWear
- **Sujeitos:** 4 (S01–S04)
- **Comandos:** `up`, `down`, `left`, `right`, `start`, `stop`, `forward`, `backward` + `rest`
- **Modos:** `silent` (silencioso) e `vocalized` (vocalizado)
- **Canais EMG:** 16 (sinais brutos e filtrados em `.h5`)

### Baixando o dataset

#### Opção 1 — Hugging Face CLI (recomendado)

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='PulpBio/SilentWear',
    repo_type='dataset',
    local_dir='./SilentWear'
)
"
```

#### Opção 2 — Git LFS

```bash
git lfs install
git clone https://huggingface.co/datasets/PulpBio/SilentWear
```

Após o download, a estrutura esperada dentro de `data_raw_and_filt/` é:

```
data_raw_and_filt/
├── S01/
│   ├── silent/
│   │   └── *.h5
│   └── vocalized/
│       └── *.h5
├── S02/
├── S03/
└── S04/
```

---

## Instalação

### Pré-requisitos

- Python 3.9+
- pip

### Dependências

```bash
pip install numpy pandas scikit-learn matplotlib umap-learn tables
```

> `tables` é necessário para leitura dos arquivos `.h5` via `pandas.read_hdf()`.

---

## Como Rodar

### 1. Configure o caminho do dataset

Abra os arquivos `trabalhofinal.py` e `reducaodimensionalidade.py` e localize o trecho abaixo no início de cada um:

```python
base_path = (
    r"C:\Users\Mille\.cache\huggingface\hub"
    r"\datasets--PulpBio--SilentWear"
    r"\snapshots\cb1f0d1fb2688c0cd0a3f21336a04e86bf906096"
    r"\data_raw_and_filt"
)
```

Substitua pelo caminho real da pasta `data_raw_and_filt/` no seu computador. Por exemplo:

**Windows:**
```python
base_path = r"C:\Users\SeuUsuario\SilentWear\data_raw_and_filt"
```

**Linux / macOS:**
```python
base_path = "/home/seuusuario/SilentWear/data_raw_and_filt"
```

> **Atenção:** Se você baixou o dataset via cache do Hugging Face, o caminho vai conter uma pasta com um hash aleatório como `cb1f0d1fb2688c0cd0a3f21336a04e86bf906096`. Esse nome **varia de máquina para máquina** e muda a cada versão do dataset. Para encontrá-la, navegue até:
> ```
> ~/.cache/huggingface/hub/datasets--PulpBio--SilentWear/snapshots/
> ```
> e use o nome da pasta que aparecer lá dentro.

### 2. Execute a classificação

```bash
python trabalhofinal.py
```

O script roda todos os experimentos automaticamente (3 classificadores × 3 cenários × 2 modos × 2 configurações de protocol cue), com validação LOSO e LOTO. Ao final, salva os resultados em:

```
classifiers_results_all_folds.csv
```

### 3. Execute a redução de dimensionalidade

```bash
python reducaodimensionalidade.py
```

Gera gráficos de PCA, UMAP e t-SNE para fala silenciosa e vocalizada, salvos como:

```
pca_umap_tsne_silent.png
pca_umap_tsne_vocalized.png
```

---

## Estrutura do Projeto

```
.
├── trabalhofinal.py           # Pipeline principal de classificação
├── reducaodimensionalidade.py # Visualização via PCA, UMAP e t-SNE
├── ARTIGO_PROJETO_...pdf      # Artigo descrevendo a metodologia e resultados
└── README.md
```

---

## Pipeline

```
Sinais EMG brutos (.h5)
        │
        ▼
Segmentação por eventos (via Label_str)
        │
        ▼
Normalização Z-score por canal (dentro do evento)
        │
        ▼
Extração de 13 features × 16 canais = vetor de 208 dimensões
        │
        ▼
Remoção de outliers (percentil 99 do max absoluto)
        │
        ▼
Padronização (StandardScaler — fitado apenas no treino)
        │
        ▼
Classificação: LDA / SVM / Random Forest
        │
        ▼
Validação: LOSO (inter-sujeito) e LOTO (intra-sujeito)
        │
        ▼
Métricas: Acurácia e Macro-F1
```

---

## Resultados

O modelo **LDA** obteve o melhor desempenho geral, atingindo **82,6% de acurácia** e **Macro-F1 de 0,660** para fala vocalizada no cenário mais completo (comandos + repouso) com validação LOTO, sem o protocol cue *start*.

Principais observações:

- **Fala vocalizada > fala silenciosa** em todos os cenários e classificadores.
- **LOTO > LOSO**: modelos generalizam bem dentro do mesmo sujeito, mas têm dificuldade com sujeitos novos — evidência de dependência do locutor.
- **Remover o sinal *start*** melhora consistentemente o desempenho, pois o protocol cue introduz variabilidade extra.
- **LDA competitivo com SVM e RF**, sugerindo que a representação de features é suficientemente discriminativa mesmo para modelos lineares simples.
- A classe `rest` é facilmente separável das demais — confirmado tanto pela redução de dimensionalidade quanto pelas acurácias quase perfeitas no experimento fala-vs-repouso.

---

## Referências

- Spacone et al. (2026). *SilentWear: an ultra-low power wearable system for EMG-based silent speech recognition.* arXiv:2603.02847
- Phinyomark et al. (2018). *Feature extraction and selection for myoelectric control based on wearable EMG sensors.* Sensors, 18(5), 1615.
- Hudgins et al. (1993). *A new strategy for multifunction myoelectric control.* IEEE Transactions on Biomedical Engineering, 40(1), 82–94.
