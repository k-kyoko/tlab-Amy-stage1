import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

project_root = os.path.abspath("/home/jovyan/work")

if project_root not in sys.path:
    sys.path.append(project_root)


def plt_corr_heatmap(
    df_results: pd.DataFrame,
    word_order: list = None,
    annot: bool = True,
    use_fdr: bool = True,
    output_filename: str = 'corr_bdi_dissim.png'
):
    """
    Generates a full symmetric heatmap from a pre-calculated correlation results DataFrame.
    Cells with p >= 0.05 are colored gray. Significant cells use a Red-White-Blue gradient.

    Parameters:
    -----------
    df_results : pd.DataFrame
        DataFrame containing the correlation results. Must contain 'Spearman_rho', 
        'p_value', 'p_value_FDR_corrected', and either a 'Word_Pair' column 
        (e.g., 'brown_anger') OR 'Word1' and 'Word2' columns.
    word_order : list, optional
        A list of words specifying the exact order.
        Any words in the list not present in the data will become NaN (blank cells).
        Any words in the data not in the list will be excluded from the plot.
    annot : bool, default True
    use_fdr : bool, default True
        If True, uses 'p_value_FDR_corrected' for the 0.05 significance threshold.
        If False, uses raw 'p_value'.
    output_filename : str
    """

    df_plot = df_results.copy()

    # Split 'Word_Pair' into 'Word1' and 'Word2' if necessary
    if 'Word1' not in df_plot.columns or 'Word2' not in df_plot.columns:
        if 'Word_Pair' in df_plot.columns:
            df_plot[['Word1', 'Word2']] = df_plot['Word_Pair'].str.split('_', n=1, expand=True)
        else:
            raise ValueError("Input DataFrame must contain either 'Word_Pair' or both 'Word1' and 'Word2' columns.")

    # Create a symmetric dataset (forward and backward directions)
    cols_to_keep = ['Word1', 'Word2', 'Spearman_rho', 'p_value', 'p_value_FDR_corrected']
    df_forward = df_plot[cols_to_keep].copy()
    df_backward = df_plot[['Word2', 'Word1', 'Spearman_rho', 'p_value', 'p_value_FDR_corrected']].copy()
    df_backward.columns = cols_to_keep

    df_sym = pd.concat([df_forward, df_backward], ignore_index=True)
    
    # 2. Pivot into Matrices
    corr_matrix = df_sym.pivot_table(index='Word1', columns='Word2', values='Spearman_rho', aggfunc='mean')
    
    pval_target = 'p_value_FDR_corrected' if use_fdr else 'p_value'
    pval_matrix = df_sym.pivot_table(index='Word1', columns='Word2', values=pval_target, aggfunc='mean')
    pval_matrix = pval_matrix.loc[corr_matrix.index, corr_matrix.columns]

    if word_order is not None:
        corr_matrix = corr_matrix.reindex(index=word_order, columns=word_order)
        pval_matrix = pval_matrix.reindex(index=word_order, columns=word_order)
    else:
        pval_matrix = pval_matrix.loc[corr_matrix.index, corr_matrix.columns]
    
    # 3. Visualization Setup
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define mask
    mask_insig = pval_matrix >= 0.05

    gray_cmap = ListedColormap(['#d3d3d3'])
    sns.heatmap(corr_matrix, cmap=gray_cmap, 
                cbar=False, annot=False, square=True, linewidths=0.5, ax=ax)
    
    # Only plots where mask_insig is False (i.e., p < 0.05).
    sns.heatmap(corr_matrix, mask=mask_insig, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                cbar=True, annot=False, square=True, linewidths=0.5, 
                cbar_kws={"shrink": .8, "label": "Spearman's rho (p<0.05)"}, ax=ax)

    if annot:
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", 
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                    cbar=False, square=True, alpha=0, ax=ax)

    sig_type = "FDR Corrected" if use_fdr else "Uncorrected"
    plt.title(f'Correlation Matrix between BDI and dissimilarity\n(p<0.05, {sig_type})', fontsize=18, pad=20)
    plt.xticks(rotation=45, fontsize=14, ha='right')
    plt.yticks(rotation=0, fontsize=14)
    plt.xlabel('')
    plt.ylabel('')
    cbar = ax.collections[1].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Spearman's rho (p<0.05)", size=14, labelpad=15)
    plt.tight_layout()

    plt.savefig(output_filename, dpi=300)


def plt_hist_winfo(
    y: np.array,
    xlabel: str,
    savepath: str = None
):

    mean_val = y.mean()
    var_val = y.var()
    std_val = y.std()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(mean_val, color='tab:blue', linestyle='dashed', linewidth=2, label='Mean')
    ax.axvspan(mean_val - std_val, mean_val + std_val, color='tab:blue', alpha=0.15, label='±1 SD')

    text_str = f'Mean: {mean_val:.3f}\nSD: {var_val:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(f"Histogram of {xlabel}", fontsize=14)
    ax.legend(loc='upper right')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plt_cumulative_sim(
    df_sim: pd.DataFrame,
    savepath: str = None,
):
    words = df_sim['Word_Pair'].unique()
    color_h1 = '#4daf4a'
    color_h0 = '#e41a1c'
    color_inc = '#cccccc'

    if savepath is None:
        save_dir = os.path.join(project_root, "outputs/figs/powercuml")
    else:
        save_dir = savepath
    
    for word in words:
        df_word = df_sim[df_sim['Word_Pair'] == word].sort_values('N')
        
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(df_word['N']))
        width = 0.6
        
        h1 = df_word['H1_Support(%)'].values
        h0 = df_word['H0_Support(%)'].values
        inc = df_word['Inconclusive(%)'].values
        
        ax.bar(x, h1, width, label='H1 Support', color=color_h1)
        ax.bar(x, h0, width, bottom=h1, label='H0 Support', color=color_h0)
        ax.bar(x, inc, width, bottom=h1+h0, label='Inconclusive', color=color_inc)
        
        ax.set_title(f"Word_Pair: {word}")
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Cumulative Percentage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(df_word['N'])
        ax.set_ylim(0, 100)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.tight_layout()

        savepath_ = os.path.join(save_dir, f"{word}.png")
        plt.savefig(savepath_)
        plt.close()