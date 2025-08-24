import pandas as pd
import numpy as np
import requests
from PIL import Image
import io
import os
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# For CNN features (optional - requires torch)
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18, ResNet18_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. CNN features will be disabled.")

def download_image(url: str, timeout: int = 10) -> Optional[Image.Image]:
    """
    Download an image from URL.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
    
    Returns:
        PIL Image object or None if download fails
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        return None

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for feature extraction.
    
    Args:
        image: PIL Image object
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image as numpy array
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    return image_array

def extract_basic_image_features(image: Image.Image) -> dict:
    """
    Extract basic image features without deep learning.
    
    Args:
        image: PIL Image object
    
    Returns:
        Dictionary with basic image features
    """
    features = {}
    
    # Basic image properties
    features['image_width'] = image.width
    features['image_height'] = image.height
    features['image_aspect_ratio'] = image.width / image.height if image.height > 0 else 0
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Color statistics
    features['mean_red'] = np.mean(img_array[:, :, 0])
    features['mean_green'] = np.mean(img_array[:, :, 1])
    features['mean_blue'] = np.mean(img_array[:, :, 2])
    features['std_red'] = np.std(img_array[:, :, 0])
    features['std_green'] = np.std(img_array[:, :, 1])
    features['std_blue'] = np.std(img_array[:, :, 2])
    
    # Overall brightness and contrast
    features['brightness'] = np.mean(img_array)
    features['contrast'] = np.std(img_array)
    
    # Color distribution
    features['red_dominance'] = features['mean_red'] / (features['mean_red'] + features['mean_green'] + features['mean_blue'])
    features['green_dominance'] = features['mean_green'] / (features['mean_red'] + features['mean_green'] + features['mean_blue'])
    features['blue_dominance'] = features['mean_blue'] / (features['mean_red'] + features['mean_green'] + features['mean_blue'])
    
    return features

def extract_cnn_features(image: Image.Image, model=None) -> np.ndarray:
    """
    Extract CNN features using a pre-trained model.
    
    Args:
        image: PIL Image object
        model: Pre-trained model (if None, uses ResNet18)
    
    Returns:
        CNN features as numpy array
    """
    if not TORCH_AVAILABLE:
        return np.array([])
    
    if model is None:
        # Load pre-trained ResNet18
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
    
    # Preprocessing for ResNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0)
    
    # Extract features
    with torch.no_grad():
        features = model(image_tensor)
        features = features.squeeze().numpy()
    
    return features

def process_listing_images(df: pd.DataFrame, image_url_col: str = 'picture_url', 
                          max_images_per_listing: int = 5, 
                          use_cnn: bool = False) -> pd.DataFrame:
    """
    Process images for all listings and extract features.
    
    Args:
        df: DataFrame with image URLs
        image_url_col: Column name containing image URLs
        max_images_per_listing: Maximum number of images to process per listing
        use_cnn: Whether to use CNN features (requires PyTorch)
    
    Returns:
        DataFrame with image features added
    """
    result_df = df.copy()
    
    print(f"🚀 Processing images for {len(df)} listings...")
    print(f"   Using CNN features: {use_cnn and TORCH_AVAILABLE}")
    
    # Initialize feature columns
    basic_features = ['image_width', 'image_height', 'image_aspect_ratio', 
                     'mean_red', 'mean_green', 'mean_blue', 'std_red', 'std_green', 'std_blue',
                     'brightness', 'contrast', 'red_dominance', 'green_dominance', 'blue_dominance']
    
    for feature in basic_features:
        result_df[f'img_{feature}'] = np.nan
    
    if use_cnn and TORCH_AVAILABLE:
        # Initialize CNN feature columns (ResNet18 has 512 features)
        for i in range(512):
            result_df[f'img_cnn_feature_{i}'] = np.nan
    
    # Process each listing
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"   Processing listing {idx}/{len(df)}")
        
        image_url = row[image_url_col]
        if pd.isna(image_url) or image_url == '':
            continue
        
        # Download image
        image = download_image(image_url)
        if image is None:
            continue
        
        # Extract basic features
        basic_feats = extract_basic_image_features(image)
        for feature, value in basic_feats.items():
            result_df.loc[idx, f'img_{feature}'] = value
        
        # Extract CNN features if requested
        if use_cnn and TORCH_AVAILABLE:
            cnn_feats = extract_cnn_features(image)
            for i, feat in enumerate(cnn_feats):
                result_df.loc[idx, f'img_cnn_feature_{i}'] = feat
    
    print("✅ Image processing complete!")
    print(f"   Basic image features: {len(basic_features)}")
    if use_cnn and TORCH_AVAILABLE:
        print(f"   CNN features: 512")
    
    return result_df

def create_image_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated image features across multiple images per listing.
    
    Args:
        df: DataFrame with image features
    
    Returns:
        DataFrame with aggregated image features
    """
    result_df = df.copy()
    
    # Get image feature columns
    img_cols = [col for col in df.columns if col.startswith('img_')]
    
    if not img_cols:
        return result_df
    
    print(f"📊 Creating aggregated image features from {len(img_cols)} image features...")
    
    # Group by listing ID and aggregate
    # For now, we'll use the first non-null value for each feature
    # In a real scenario, you might want to aggregate multiple images per listing
    
    for col in img_cols:
        # Fill missing values with median
        median_val = result_df[col].median()
        result_df[col] = result_df[col].fillna(median_val)
    
    print("✅ Image aggregation complete!")
    
    return result_df

def create_image_features(df: pd.DataFrame, use_cnn: bool = False) -> pd.DataFrame:
    """
    Create comprehensive image features for the dataset.
    
    Args:
        df: DataFrame with image URLs
        use_cnn: Whether to use CNN features (requires PyTorch)
    
    Returns:
        DataFrame with all image features added
    """
    print("🚀 Creating image features...")
    
    # Process images
    result_df = process_listing_images(df, use_cnn=use_cnn)
    
    # Create aggregated features
    result_df = create_image_aggregation_features(result_df)
    
    print("✅ Image features created successfully!")
    print(f"   Original shape: {df.shape}")
    print(f"   New shape: {result_df.shape}")
    print(f"   New image features: {result_df.shape[1] - df.shape[1]}")
    
    return result_df
