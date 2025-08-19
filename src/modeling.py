# In src/modeling.py

import pymc as pm
import numpy as np
import pandas as pd

def fit_beta_prior(league_data: pd.DataFrame) -> tuple[float, float]:
    """
    Fits a Beta distribution to the league's HR rates
    using the method of moments.
    """
    # Calculate mean and variance of the observed HR rates
    mu = league_data['HR_RATE'].mean()
    var = league_data['HR_RATE'].var()
    
    # Calculate alpha and beta via method of moments
    alpha = mu * ((mu * (1 - mu) / var) - 1)
    beta = (1 - mu) * ((mu * (1 - mu) / var) - 1)
    
    return alpha, beta

def run_beta_binomial_model(player_stats: dict, prior_alpha: float, prior_beta: float):
    """
    Runs a Beta-Binomial MCMC simulation for a player's HR rate.
    
    Args:
        player_stats (dict): Dict with 'HR' and 'PA' keys for the player.
        prior_alpha (float): Alpha parameter for the Beta prior.
        prior_beta (float): Beta parameter for the Beta prior.
        
    Returns:
        A PyMC trace object (idata).
    """
    with pm.Model() as model:
        # Prior for the player's HR rate, informed by the league
        theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
        
        # Likelihood of observing the player's career stats
        y = pm.Binomial('y', 
                        n=player_stats['PA'], 
                        p=theta, 
                        observed=player_stats['HR'])
        
        # Run the MCMC simulation
        idata = pm.sample(4000, tune=2000, chains=4, cores=1, progressbar=True)
        
    return idata

def simulate_season(idata, future_pa: int) -> np.ndarray:
    """
    Simulates season HR totals from a posterior trace.
    
    Args:
        idata: The PyMC trace object from the model.
        future_pa (int): The estimated number of plate appearances for the season.
        
    Returns:
        An array of simulated home run totals.
    """
    # Extract the posterior samples for theta
    posterior_theta = idata.posterior['theta'].values.flatten()
    
    # Simulate future HRs for each theta sample
    simulated_hrs = np.random.binomial(n=future_pa, p=posterior_theta)
    
    return simulated_hrs