# -*- coding: utf-8 -*-
"""
Analysis script for the paper "From Populus Romanus to Populus Christianus:
The Concept of 'People' in Thomas Aquinas in Light of Distributional Semantics".

This script downloads historical Latin word embedding models, performs a diachronic
semantic analysis, and generates all figures and tables used in the study.

To run:
1. Ensure all libraries from requirements.txt are installed.
2. Execute from the root directory of the project: python src/analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import KeyedVectors
from numpy.linalg import norm
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore")

# ==================================================
# 1. CONFIGURATION
# ==================================================

# --- File Paths and URLs ---
DATA_DIR = "data"
RESULTS_DIR = "results"

CLASSICAL_MODEL_URL = 'https://embeddings.lila-erc.eu/samples/download/aligned/OperaLatina.vec.txt'
AQUINAS_MODEL_URL = 'https://embeddings.lila-erc.eu/samples/download/aligned/OperaMaiora.vec.txt'

CLASSICAL_FILE = os.path.join(DATA_DIR, "OperaLatina.vec.txt")
AQUINAS_FILE = os.path.join(DATA_DIR, "OperaMaiora.vec.txt")

# --- Word Lists for Analysis ---
# Note: Words are normalized with 'u' instead of 'v' as per the LiLa models' convention.
CORE_CONCEPTS = [
    "populus", "plebs", "gens", "natio", "uulgus", "multitudo"
]

SEMANTIC_ANCHORS = [
    "ciuitas", "ciuis", "patricius", "respublica", "lex",
    "genus", "stirps", "familia", "regnum", "rex", "princeps",
    "imperium", "fides", "ecclesia", "christianus", "deus",
    "lingua", "turba", "tumultus"
]

# Words selected for heatmap visualization
HEATMAP_WORDS = sorted(list(set(CORE_CONCEPTS + [
    "ecclesia", "ciuitas", "patricius", "fides", "lex",
    "ciuis", "respublica", "princeps", "christianus"
])))

# Combined list of all words to be analyzed
ALL_ANALYSIS_WORDS = sorted(list(set(CORE_CONCEPTS + SEMANTIC_ANCHORS + HEATMAP_WORDS)))


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def download_if_missing(url: str, filename: str):
    """Downloads a file from a URL if it doesn't already exist."""
    if not os.path.exists(filename):
        print(f"Downloading {os.path.basename(filename)}...")
        os.system(f"wget -O {filename} {url}")
    else:
        print(f"{os.path.basename(filename)} already exists, skipping download.")

def load_model(filename: str) -> KeyedVectors:
    """Loads a word2vec model from a text file."""
    print(f"Loading embeddings from {os.path.basename(filename)}...")
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    print(f"Loaded. Vocabulary size: {len(model.key_to_index)}")
    return model

def get_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def get_vectors_for_words(model: KeyedVectors, words: list) -> (np.ndarray, list):
    """Extracts vectors for a given list of words from a model."""
    vectors, valid_words = [], []
    for w in words:
        if w in model:
            vectors.append(model[w])
            valid_words.append(w)
    return np.array(vectors), valid_words

def plot_projection(coords_cl: np.ndarray, coords_aq: np.ndarray, words: list, title: str, filename: str):
    """Generates and saves a 2-panel PCA plot."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    def plot_single(ax, coords, plot_title):
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
        for i, txt in enumerate(words):
            color = 'red' if txt in CORE_CONCEPTS else 'black'
            weight = 'bold' if txt in CORE_CONCEPTS else 'normal'
            ax.annotate(txt, (coords[i, 0], coords[i, 1]), color=color, fontweight=weight, fontsize=12)
        ax.set_title(plot_title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)

    plot_single(axes[0], coords_cl, "Classical Latin (Opera Latina)")
    plot_single(axes[1], coords_aq, "Aquinas (Opera Maiora)")
    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300)
    print(f"Projection plot saved as '{filename}'")
    plt.show()


# ==================================================
# 3. MAIN ANALYSIS PIPELINE
# ==================================================

def main():
    """Main function to run the complete analysis pipeline."""
    # --- 3.1 Setup and Model Loading ---
    print("="*50)
    print("PHASE 1: SETUP AND MODEL LOADING")
    print("="*50)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    download_if_missing(CLASSICAL_MODEL_URL, CLASSICAL_FILE)
    download_if_missing(AQUINAS_MODEL_URL, AQUINAS_FILE)

    model_classical = load_model(CLASSICAL_FILE)
    model_aquinas = load_model(AQUINAS_FILE)
    common_vocab = list(set(model_classical.key_to_index) & set(model_aquinas.key_to_index))
    print(f"\nCommon vocabulary size: {len(common_vocab)}")

    # --- 3.2 Global Semantic Shift Analysis ---
    print("\n" + "="*50)
    print("PHASE 2: GLOBAL SEMANTIC SHIFT ANALYSIS")
    print("="*50)

    print("Computing cosine similarities for the full shared vocabulary...")
    global_sims = [
        (lemma, get_cosine_similarity(model_classical[lemma], model_aquinas[lemma]))
        for lemma in tqdm(common_vocab, desc="Global Shift Calculation")
    ]
    df_global = pd.DataFrame(global_sims, columns=['word', 'similarity'])
    df_global.to_csv(os.path.join(RESULTS_DIR, 'global_semantic_shifts.csv'), index=False)
    cos_values = df_global['similarity'].values

    # Calculate statistics for shift thresholds
    mu, sigma = cos_values.mean(), cos_values.std(ddof=1)
    median = np.median(cos_values)
    print("\n--- Global Shift Statistics ---")
    print(f"Mean (μ): {mu:.4f}")
    print(f"Median:   {median:.4f}")
    print(f"Std Dev (σ):  {sigma:.4f}")
    print(f"Range [μ ± 1σ]: [{mu - sigma:.4f}, {mu + sigma:.4f}]")
    print(f"Range [μ ± 2σ]: [{mu - 2*sigma:.4f}, {mu + 2*sigma:.4f}]")

    # Top 10 most shifted and stable words
    df_sorted = df_global.sort_values('similarity')
    print("\nTop 10 Most Shifted (Lowest Similarity):")
    print(df_sorted.head(10).to_string(index=False))
    print("\nTop 10 Most Stable (Highest Similarity):")
    print(df_sorted.tail(10).iloc[::-1].to_string(index=False))

    # Plot and save histogram of global shifts
    plt.figure(figsize=(12, 7))
    plt.hist(cos_values, bins=60, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mu, color='green', linewidth=2, label=f'Mean (μ): {mu:.3f}')
    plt.axvline(mu - sigma, color='blue', linestyle='--', label=f'μ ± 1σ')
    plt.axvline(mu + sigma, color='blue', linestyle='--')
    plt.axvline(mu - 2*sigma, color='red', linestyle='-', label=f'μ ± 2σ')
    plt.axvline(mu + 2*sigma, color='red', linestyle='-')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Number of Words')
    plt.title('Distribution of Semantic Shift (Classical Latin vs. Aquinas)')
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(RESULTS_DIR, 'semantic_shift_histogram.png')
    plt.savefig(hist_path, dpi=300)
    print(f"\nHistogram saved as '{hist_path}'")
    plt.show()

    # --- 3.3 Focused Analysis for Key Terms ---
    print("\n" + "="*50)
    print("PHASE 3: FOCUSED ANALYSIS OF KEY TERMS")
    print("="*50)

    focused_sims = [
        (word, get_cosine_similarity(model_classical[word], model_aquinas[word]))
        for word in ALL_ANALYSIS_WORDS if word in common_vocab
    ]
    df_focused = pd.DataFrame(focused_sims, columns=['word', 'similarity']).sort_values('similarity')
    df_focused.to_csv(os.path.join(RESULTS_DIR, 'focused_terms_shifts.csv'), index=False)
    print("\nSemantic Shift for Selected Terms:")
    print(df_focused.to_string(index=False))

    # Plot and save bar chart for focused terms
    plt.figure(figsize=(12, 10))
    colors = ['#d62728' if x < mu else '#7f7f7f' for x in df_focused['similarity']]
    bars = plt.barh(df_focused['word'], df_focused['similarity'], color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.axvline(mu, color='green', linestyle='--', label=f'Mean Shift: {mu:.3f}')
    plt.title('Semantic Similarity Shift: Classical vs. Aquinas (Selected Terms)')
    plt.xlabel('Cosine Similarity')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.legend()
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width >= 0 else width - 0.06
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center')
    plt.tight_layout()
    barchart_path = os.path.join(RESULTS_DIR, 'focused_terms_barchart.png')
    plt.savefig(barchart_path, dpi=300)
    print(f"\nBar chart saved as '{barchart_path}'")
    plt.show()

    # --- 3.4 Heatmaps of Pairwise Similarities ---
    print("\n" + "="*50)
    print("PHASE 4: PAIRWISE SIMILARITY HEATMAPS")
    print("="*50)
    print(f"Building heatmaps for {len(HEATMAP_WORDS)} words...")
    vecs_cl, valid_words = get_vectors_for_words(model_classical, HEATMAP_WORDS)
    vecs_aq, _ = get_vectors_for_words(model_aquinas, HEATMAP_WORDS)

    sim_cl = cosine_similarity(vecs_cl)
    sim_aq = cosine_similarity(vecs_aq)
    diff_matrix = sim_aq - sim_cl

    heatmap_titles = ["Pairwise Similarity: Classical Latin", "Pairwise Similarity: Aquinas", "Change in Similarity (Aquinas - Classical)"]
    heatmap_data = [sim_cl, sim_aq, diff_matrix]
    heatmap_cmaps = ["viridis", "viridis", "coolwarm"]
    heatmap_filenames = ["heatmap_classical.png", "heatmap_aquinas.png", "heatmap_difference.png"]

    for title, data, cmap, fname in zip(heatmap_titles, heatmap_data, heatmap_cmaps, heatmap_filenames):
        plt.figure(figsize=(12, 10))
        sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, center=0 if "Change" in title else None,
                    xticklabels=valid_words, yticklabels=valid_words)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        fpath = os.path.join(RESULTS_DIR, fname)
        plt.savefig(fpath, dpi=300)
        print(f"Heatmap '{fpath}' saved.")
        plt.show()

    # --- 3.5 PCA Projection ---
    print("\n" + "="*50)
    print("PHASE 5: PCA PROJECTION OF SEMANTIC SPACE")
    print("="*50)
    words_viz = [w for w in ALL_ANALYSIS_WORDS if w in common_vocab]
    v_cl_viz, _ = get_vectors_for_words(model_classical, words_viz)
    v_aq_viz, _ = get_vectors_for_words(model_aquinas, words_viz)

    if len(words_viz) > 2:
        combined_vectors = np.vstack([v_cl_viz, v_aq_viz])
        pca = PCA(n_components=2, random_state=42)
        coords_pca = pca.fit_transform(combined_vectors)

        coords_cl_pca = coords_pca[:len(words_viz)]
        coords_aq_pca = coords_pca[len(words_viz):]

        pca_path = os.path.join(RESULTS_DIR, 'pca_projection.png')
        plot_projection(coords_cl_pca, coords_aq_pca, words_viz,
                        "PCA Projection of Semantic Spaces", pca_path)

    # --- 3.6 Nearest Neighbors Analysis ---
    print("\n" + "="*50)
    print("PHASE 6: NEAREST NEIGHBORS ANALYSIS")
    print("="*50)
    report_path = os.path.join(RESULTS_DIR, 'nearest_neighbors_report.txt')
    with open(report_path, "w", encoding="utf-8") as f:
        print(f"Generating nearest neighbors report at '{report_path}'...")
        for word in CORE_CONCEPTS:
            if word in model_classical and word in model_aquinas:
                header = f"\n=== {word.upper()} ==="
                print(header); f.write(header + "\n")

                sims_cl = model_classical.most_similar(word, topn=15)
                sims_aq = model_aquinas.most_similar(word, topn=15)

                row_fmt = "{:<25} {:<10} | {:<25} {:<10}"
                table_header = row_fmt.format("Classical Neighbor", "Sim", "Aquinas Neighbor", "Sim")
                print(table_header); f.write(table_header + "\n")
                print("-" * 75); f.write("-" * 75 + "\n")

                for i in range(15):
                    cl_neighbor, cl_sim = sims_cl[i]
                    aq_neighbor, aq_sim = sims_aq[i]
                    line = row_fmt.format(cl_neighbor, f"{cl_sim:.4f}", aq_neighbor, f"{aq_sim:.4f}")
                    print(line); f.write(line + "\n")

    print("\n" + "="*50)
    print("FULL ANALYSIS COMPLETE.")
    print(f"All results saved in '{RESULTS_DIR}' directory.")
    print("="*50)

if __name__ == "__main__":
    main()