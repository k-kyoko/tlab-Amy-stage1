import os
import sys
import numpy as np
import math
import pandas as pd
import pingouin as pg

project_root = os.path.abspath("/home/jovyan/work")
if project_root not in sys.path:
    sys.path.append(project_root)


def sim_df10(
    df: pd.DataFrame,
    N_list: list = [100, 200, 400, 800, 1000],   # Sample sizes to test
    M: int = 100,                                # Number of bootstrap iterations
    threshold_H1: float = 6.0,                   # Threshold supporting H1 (Correlation exists)
    threshold_H0: float = 1 / 6.0,               # Threshold supporting H0 (No correlation)
    target_col: str = 'BDI',
) -> pd.DataFrame:

    '''
    Simulate DF10 based on the DataFrame (Contains rows named "Sub" and target col ("BDI"), word pairs)
    Using boot strap method
    '''
    
    word_cols = [col for col in df.columns if col not in ['Sub', target_col]]
    simulation_results = []
    
    print("Starting simulation...")
    
    for N in N_list:
        counts = {word: {'H1': 0, 'H0': 0, 'Inc': 0} for word in word_cols}
        
        for m in range(M):
            # Bootstrap sampling and Calculate Spearman's rho
            df_boot = df.sample(n=N, replace=True)
            rho_series = df_boot[word_cols].corrwith(df_boot[target_col], method='spearman')
            
            rho_values = rho_series.fillna(0).values
            rho_values = np.clip(rho_values, -0.999, 0.999)
            
            # Calculate BF10 and Count thresholds per word pair
            for i, word in enumerate(word_cols):
                r = rho_values[i]
                bf10 = pg.bayesfactor_pearson(r, n=N)
                
                if bf10 >= threshold_H1:
                    counts[word]['H1'] += 1
                elif bf10 <= threshold_H0:
                    counts[word]['H0'] += 1
                else:
                    counts[word]['Inc'] += 1
    
        # Calculate percentages
        for word in word_cols:
            simulation_results.append({
                'Word_Pair': word,
                'N': N,
                'H1_Support(%)': (counts[word]['H1'] / M) * 100,
                'H0_Support(%)': (counts[word]['H0'] / M) * 100,
                'Inconclusive(%)': (counts[word]['Inc'] / M) * 100
            })
            
        print(f"Completed processing for N={N}")
    
    df_sim = pd.DataFrame(simulation_results)

    return df_sim