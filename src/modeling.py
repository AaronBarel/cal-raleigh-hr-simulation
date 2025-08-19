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

def run_hierarchical_model(data, features):
    """
    Fits a hierarchical Bayesian model to player data using PyMC.

    Args:
        data (pd.DataFrame): DataFrame containing player data (HR, PA, features).
        features (list): List of feature names to use as predictors.

    Returns:
        pm.InferenceData: The InferenceData object containing the posterior draws.
    """
    coords = {"player": data.index.values, "features": features}

    with pm.Model(coords=coords) as hierarchical_model:
        # Population-level parameters (hyperpriors)
        mu_alpha = pm.Normal('mu_alpha', mu=-2.5, sigma=1)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)

        # Regression coefficients for advanced stats
        betas = pm.Normal('betas', mu=0, sigma=0.5, dims='features')

        # Player-level random effects (hierarchical priors)
        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, dims="player")

        # The log-odds of a player's true HR rate
        theta_logits = alpha + pm.math.dot(data[features].values, betas)

        # Convert from log-odds to probability
        theta = pm.Deterministic('theta', pm.math.invlogit(theta_logits))

        # Likelihood (the data) - CORRECTED TO USE 'PA_target' AND 'HR_target'
        pm.Binomial('likelihood', n=data['PA_target'], p=theta, observed=data['HR_target'])

        idata = pm.sample(2000, tune=2000, cores=2)

    return idata