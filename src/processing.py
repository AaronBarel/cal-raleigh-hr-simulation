# src/processing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def load_data(filepath='../data/raw/fangraphs-leaderboards.csv'):
    """Loads the raw FanGraphs data."""
    return pd.read_csv(filepath)

def get_player_career_stats(df, player_name, end_year=2024):
    """Calculates weighted career stats for a single player up to a given year."""
    player_df = df[(df['Name'] == player_name) & (df['Season'] <= end_year)].copy()
    
    total_pa = player_df['PA'].sum()
    total_hr = player_df['HR'].sum()
    
    if total_pa == 0:
        return None

    stats = {
        'Name': player_name,
        'PA': total_pa,
        'HR': total_hr,
        'HR_RATE': total_hr / total_pa,
    }
    
    for col in ['BB%', 'SLG', 'ISO', 'Med%', 'Hard%', 'Barrel%', 'HardHit%']:
        weighted_avg = (player_df[col] * player_df['PA']).sum() / total_pa
        stats[col] = weighted_avg
        
    return stats

def find_similar_players_knn(df, target_player_name, features, n_neighbors=10):
    """Finds similar players using k-Nearest Neighbors."""
    
    # Get all player career stats up to 2024
    player_names = df[df['Season'] < 2025]['Name'].unique()
    career_stats_list = [get_player_career_stats(df, name) for name in player_names]
    career_df = pd.DataFrame([s for s in career_stats_list if s is not None and s['PA'] > 500]) # Min PA for stability
    
    # Prepare the data for KNN
    X = career_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree') # +1 to include the player himself
    knn.fit(X_scaled)
    
    # Find the target player and get his neighbors
    target_idx = career_df[career_df['Name'] == target_player_name].index
    if len(target_idx) == 0:
        raise ValueError(f"{target_player_name} not found in career stats DataFrame.")
        
    target_scaled = X_scaled[target_idx]
    distances, indices = knn.kneighbors(target_scaled)
    
    # Get the names of the similar players (excluding the player himself)
    similar_player_indices = indices.flatten()[1:]
    similar_players_df = career_df.iloc[similar_player_indices]
    
    return similar_players_df

def create_panel_data_for_modeling(df, start_year=2015, end_year=2024):
    """
    Restructures data to create a 3-year history -> 1-year outcome format.
    This is perfect for the hierarchical model.
    """
    df_filtered = df[(df['Season'] >= start_year) & (df['Season'] <= end_year) & (df['PA'] >= 250)]
    
    # Use lag to get previous seasons' stats
    df_filtered = df_filtered.sort_values(['Name', 'Season'])
    for col in ['PA', 'HR', 'Barrel%', 'HardHit%', 'ISO']:
        for i in range(1, 4): # Lags for t-1, t-2, t-3
            df_filtered[f'{col}_lag{i}'] = df_filtered.groupby('Name')[col].shift(i)

    # The target variable is the current season's HR and PA
    df_filtered.rename(columns={'HR': 'HR_target', 'PA': 'PA_target'}, inplace=True)
    
    # Drop rows with missing lagged data (i.e., players without 3 prior seasons)
    final_df = df_filtered.dropna()
    
    return final_df