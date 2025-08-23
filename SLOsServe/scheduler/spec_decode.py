from dataclasses import dataclass
import numpy as np
from typing import List 
from scipy.stats import geom, chisquare
import os

@dataclass
class SpecDecode:
    alpha: float
    max_spec_decode: int
        
    def fit(self, data : List[int], prefix: str | None = None):
        # Observed data
        max_acc = max(data)
        data = [x for x in data if x < max_acc]
        # Observed frequencies
        observed_counts = np.bincount(data)[1:]  # Exclude 0, as geom starts at 1
        print('observed_counts', observed_counts)
        n = sum(observed_counts)
        # Estimate p
        p_estimated = 1 / np.mean(data)
        self.alpha = 1 - float(p_estimated)

        # Expected frequencies
        expected_counts = [n * geom.pmf(k, p_estimated) for k in range(1, len(observed_counts) + 1)]
        print(sum(expected_counts), sum(observed_counts))
        
        n_exp = sum(expected_counts)
        
        observed_counts = observed_counts.astype(np.float64) * n_exp / n
            
        
        print('expected_counts', expected_counts)
        print('estimated alpha', self.alpha)

        # Chi-square test
        chi2_stat, p_value = chisquare(observed_counts, f_exp=expected_counts)
        
        
        print("Chi-Square Statistic:", chi2_stat)
        print("p-value:", p_value)
        
        if prefix is not None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(tight_layout = True)
            # Plot observed frequencies
            ax.bar(range(1, len(observed_counts) + 1), observed_counts / n, alpha=0.6, color='g', label='Observed')
            ax.set_ylabel('Probability', fontsize = 12)
            ax.set_xlabel('#Acc', fontsize = 12)
            # Plot expected frequencies
            x = np.arange(1, len(observed_counts) + 1)
            ax.plot(x, geom.pmf(x, p_estimated), 'ro-', label='Expected')
            
            plt.legend()
            fig.savefig(f'{prefix}-spec-profile.png')
            fig.savefig(f'{prefix}-spec-profile.pdf')
            print(f'spec decode fitting is saved to {prefix}-spec-profile.png')
            
        sample_variance = np.var(data, ddof=1)
        expected_variance = (1 - p_estimated) / (p_estimated ** 2)

        print("Sample Variance:", sample_variance)
        print("Expected Variance:", expected_variance)
        
    def exp(self, bs: int) -> float:
        return (1 - self.alpha ** bs) / (1 - self.alpha)
    
    def sample(self, bs: int) -> int:
        return min(np.random.geometric(1 - self.alpha), bs)