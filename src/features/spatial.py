import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def calculate_haversine_distance(df: pd.DataFrame, lat1_col: str = 'latitude', 
                                lon1_col: str = 'longitude', 
                                lat2: float = None, lon2: float = None) -> pd.DataFrame:
    """
    Calculate haversine distance between points and a reference point.
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat1_col: Name of latitude column
        lon1_col: Name of longitude column
        lat2: Reference latitude (if None, uses city center)
        lon2: Reference longitude (if None, uses city center)
    
    Returns:
        DataFrame with distance features added
    """
    result_df = df.copy()
    
    # Default to Munich city center if no reference point provided
    if lat2 is None:
        lat2 = 48.137154  # Munich city center
    if lon2 is None:
        lon2 = 11.576124  # Munich city center
    
    # Vectorized calculation using cdist
    coords = result_df[[lat1_col, lon1_col]].values
    ref_coords = np.array([[lat2, lon2]])
    distances = cdist(coords, ref_coords, metric='euclidean').flatten() * 111  # Convert to km
    
    result_df['distance_to_center'] = distances
    
    return result_df

def create_spatial_clusters(df: pd.DataFrame, n_clusters: int = 10, 
                          method: str = 'kmeans') -> pd.DataFrame:
    """
    Create spatial clusters using K-means or DBSCAN.
    
    Args:
        df: DataFrame with latitude and longitude columns
        n_clusters: Number of clusters for K-means
        method: 'kmeans' or 'dbscan'
    
    Returns:
        DataFrame with cluster labels added
    """
    result_df = df.copy()
    
    # Prepare coordinates
    coords = result_df[['latitude', 'longitude']].dropna()
    
    if method == 'kmeans':
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Add cluster labels back to original dataframe
        result_df.loc[coords.index, 'spatial_cluster'] = cluster_labels
        
        # Calculate cluster centers
        cluster_centers = kmeans.cluster_centers_
        result_df['cluster_center_lat'] = result_df['spatial_cluster'].map(
            {i: center[0] for i, center in enumerate(cluster_centers)}
        )
        result_df['cluster_center_lon'] = result_df['spatial_cluster'].map(
            {i: center[1] for i, center in enumerate(cluster_centers)}
        )
        
    elif method == 'dbscan':
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.01, min_samples=5)  # Adjust parameters as needed
        cluster_labels = dbscan.fit_predict(coords)
        
        # Add cluster labels back to original dataframe
        result_df.loc[coords.index, 'spatial_cluster'] = cluster_labels
    
    return result_df

def calculate_point_of_interest_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distances to important points of interest in Munich.
    
    Args:
        df: DataFrame with latitude and longitude columns
    
    Returns:
        DataFrame with POI distance features added
    """
    result_df = df.copy()
    
    # Important POIs in Munich (latitude, longitude)
    pois = {
        'marienplatz': (48.137154, 11.576124),
        'oktoberfest': (48.1315, 11.5498),
        'english_garden': (48.1589, 11.6156),
        'olympic_park': (48.1758, 11.5497),
        'airport': (48.3538, 11.7861),
        'central_station': (48.1402, 11.5560),
        'university': (48.1508, 11.5805),
        'bmw_welt': (48.1773, 11.5597)
    }
    
    # Vectorized calculation for all POIs
    coords = result_df[['latitude', 'longitude']].values
    
    for poi_name, (poi_lat, poi_lon) in pois.items():
        # Calculate distances using vectorized operation
        poi_coords = np.array([[poi_lat, poi_lon]])
        distances = cdist(coords, poi_coords, metric='euclidean').flatten() * 111  # Convert to km
        result_df[f'distance_to_{poi_name}'] = distances
    
    return result_df

def create_density_features(df: pd.DataFrame, radius_km: float = 1.0) -> pd.DataFrame:
    """
    Create density features based on nearby listings (optimized version).
    
    Args:
        df: DataFrame with latitude and longitude columns
        radius_km: Radius in kilometers to consider for density calculation
    
    Returns:
        DataFrame with density features added
    """
    result_df = df.copy()
    
    # Use vectorized operations for better performance
    
    # Get coordinates
    coords = result_df[['latitude', 'longitude']].values
    
    # Calculate pairwise distances using vectorized operation
    # Convert to approximate km (1 degree ≈ 111 km)
    distances_km = cdist(coords, coords) * 111
    
    # Create density features
    density_features = []
    avg_price_features = []
    
    for i in range(len(result_df)):
        # Get distances from current point to all others
        distances = distances_km[i]
        
        # Find nearby listings (within radius, excluding self)
        nearby_mask = (distances <= radius_km) & (distances > 0)
        
        # Count nearby listings
        density = np.sum(nearby_mask)
        density_features.append(density)
        
        # Average price of nearby listings
        if density > 0:
            nearby_prices = result_df.loc[nearby_mask, 'price'].values
            avg_price = np.mean(nearby_prices)
        else:
            avg_price = np.nan
        avg_price_features.append(avg_price)
    
    result_df[f'listings_density_{radius_km}km'] = density_features
    result_df[f'avg_price_{radius_km}km'] = avg_price_features
    
    return result_df

def create_fast_spatial_features(df: pd.DataFrame, max_samples: int = 5000) -> pd.DataFrame:
    """
    Create spatial features optimized for large datasets.
    Skips density features for datasets larger than max_samples.
    
    Args:
        df: DataFrame with latitude and longitude columns
        max_samples: Maximum number of samples for density calculation
    
    Returns:
        DataFrame with spatial features added
    """
    print("🚀 Creating fast spatial features...")
    
    # 1. Distance to city center
    print("1. Calculating distance to city center...")
    result_df = calculate_haversine_distance(df)
    
    # 2. Spatial clustering
    print("2. Creating spatial clusters...")
    result_df = create_spatial_clusters(result_df, n_clusters=10, method='kmeans')
    
    # 3. POI distances
    print("3. Calculating distances to points of interest...")
    result_df = calculate_point_of_interest_distances(result_df)
    
    # 4. Density features (only for smaller datasets)
    if len(df) <= max_samples:
        print("4. Creating density features...")
        result_df = create_density_features(result_df, radius_km=1.0)
        result_df = create_density_features(result_df, radius_km=2.0)
    else:
        print(f"4. Skipping density features (dataset too large: {len(df)} > {max_samples})")
        # Add placeholder columns
        result_df['listings_density_1.0km'] = np.nan
        result_df['avg_price_1.0km'] = np.nan
        result_df['listings_density_2.0km'] = np.nan
        result_df['avg_price_2.0km'] = np.nan
    
    # 5. Additional spatial features
    print("5. Creating additional spatial features...")
    
    # Distance to nearest listing (only for smaller datasets)
    if len(df) <= max_samples:
        coords = result_df[['latitude', 'longitude']].values
        distances_km = cdist(coords, coords) * 111
        
        # Find minimum distance to other listings (excluding self)
        distances_km_no_self = distances_km.copy()
        np.fill_diagonal(distances_km_no_self, np.inf)
        min_distances = np.min(distances_km_no_self, axis=1)
        min_distances[min_distances == np.inf] = np.nan
        
        result_df['distance_to_nearest_listing'] = min_distances
    else:
        result_df['distance_to_nearest_listing'] = np.nan
    
    # Spatial coordinates as features (normalized)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(result_df[['latitude', 'longitude']].fillna(0))
    result_df['latitude_scaled'] = coords_scaled[:, 0]
    result_df['longitude_scaled'] = coords_scaled[:, 1]
    
    print("✅ Fast spatial features created successfully!")
    print(f"   Original shape: {df.shape}")
    print(f"   New shape: {result_df.shape}")
    print(f"   New spatial features: {result_df.shape[1] - df.shape[1]}")
    
    return result_df

def create_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive spatial features for the dataset.
    
    Args:
        df: DataFrame with latitude and longitude columns
    
    Returns:
        DataFrame with all spatial features added
    """
    print("🚀 Creating spatial features...")
    
    # 1. Distance to city center
    print("1. Calculating distance to city center...")
    result_df = calculate_haversine_distance(df)
    
    # 2. Spatial clustering
    print("2. Creating spatial clusters...")
    result_df = create_spatial_clusters(result_df, n_clusters=10, method='kmeans')
    
    # 3. POI distances
    print("3. Calculating distances to points of interest...")
    result_df = calculate_point_of_interest_distances(result_df)
    
    # 4. Density features
    print("4. Creating density features...")
    result_df = create_density_features(result_df, radius_km=1.0)
    result_df = create_density_features(result_df, radius_km=2.0)
    
    # 5. Additional spatial features
    print("5. Creating additional spatial features...")
    
    # Distance to nearest listing (using the same distance matrix)
    coords = result_df[['latitude', 'longitude']].values
    distances_km = cdist(coords, coords) * 111
    
    # Find minimum distance to other listings (excluding self)
    distances_km_no_self = distances_km.copy()
    np.fill_diagonal(distances_km_no_self, np.inf)
    min_distances = np.min(distances_km_no_self, axis=1)
    min_distances[min_distances == np.inf] = np.nan
    
    result_df['distance_to_nearest_listing'] = min_distances
    
    # Spatial coordinates as features (normalized)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(result_df[['latitude', 'longitude']].fillna(0))
    result_df['latitude_scaled'] = coords_scaled[:, 0]
    result_df['longitude_scaled'] = coords_scaled[:, 1]
    
    print("✅ Spatial features created successfully!")
    print(f"   Original shape: {df.shape}")
    print(f"   New shape: {result_df.shape}")
    print(f"   New spatial features: {result_df.shape[1] - df.shape[1]}")
    
    return result_df
