import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, List, Optional

def extract_sentiment_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Extract sentiment features from text columns using VADER sentiment analysis.
    
    Args:
        df: DataFrame containing the text data
        text_column: Name of the column containing text to analyze
        
    Returns:
        DataFrame with sentiment features added
    """
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Extract sentiment scores
    sentiment_scores = []
    for text in result_df[text_column].fillna(''):
        if isinstance(text, str) and text.strip():
            scores = sia.polarity_scores(text)
            sentiment_scores.append(scores)
        else:
            sentiment_scores.append({'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
    
    # Add sentiment features
    result_df[f'{text_column}_negative'] = [score['neg'] for score in sentiment_scores]
    result_df[f'{text_column}_neutral'] = [score['neu'] for score in sentiment_scores]
    result_df[f'{text_column}_positive'] = [score['pos'] for score in sentiment_scores]
    result_df[f'{text_column}_compound'] = [score['compound'] for score in sentiment_scores]
    
    return result_df

def extract_text_length_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Extract text length and basic text features.
    
    Args:
        df: DataFrame containing the text data
        text_column: Name of the column containing text
        
    Returns:
        DataFrame with text length features added
    """
    result_df = df.copy()
    
    # Text length features
    result_df[f'{text_column}_length'] = result_df[text_column].fillna('').str.len()
    result_df[f'{text_column}_word_count'] = result_df[text_column].fillna('').str.split().str.len()
    result_df[f'{text_column}_char_count_no_spaces'] = result_df[text_column].fillna('').str.replace(' ', '').str.len()
    
    # Average word length
    result_df[f'{text_column}_avg_word_length'] = (
        result_df[f'{text_column}_char_count_no_spaces'] / 
        result_df[f'{text_column}_word_count'].replace(0, 1)
    )
    
    return result_df

def extract_text_complexity_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Extract text complexity features like punctuation, capitalization, etc.
    
    Args:
        df: DataFrame containing the text data
        text_column: Name of the column containing text
        
    Returns:
        DataFrame with text complexity features added
    """
    result_df = df.copy()
    
    text_series = result_df[text_column].fillna('')
    
    # Punctuation features
    result_df[f'{text_column}_exclamation_count'] = text_series.str.count('!')
    result_df[f'{text_column}_question_count'] = text_series.str.count('?')
    result_df[f'{text_column}_period_count'] = text_series.str.count(r'\.')
    result_df[f'{text_column}_comma_count'] = text_series.str.count(',')
    
    # Capitalization features
    result_df[f'{text_column}_uppercase_count'] = text_series.str.count(r'[A-Z]')
    result_df[f'{text_column}_lowercase_count'] = text_series.str.count(r'[a-z]')
    result_df[f'{text_column}_uppercase_ratio'] = (
        result_df[f'{text_column}_uppercase_count'] / 
        (result_df[f'{text_column}_uppercase_count'] + result_df[f'{text_column}_lowercase_count']).replace(0, 1)
    )
    
    # Special characters
    result_df[f'{text_column}_special_char_count'] = text_series.str.count(r'[^a-zA-Z0-9\s]')
    
    return result_df

def process_reviews_sentiment(listings_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process reviews and add sentiment features to listings.
    
    Args:
        listings_df: DataFrame with listings data
        reviews_df: DataFrame with reviews data
        
    Returns:
        DataFrame with aggregated review sentiment features
    """
    # Extract sentiment from reviews
    reviews_with_sentiment = extract_sentiment_features(reviews_df, 'comments')
    
    # Aggregate sentiment by listing_id
    sentiment_agg = reviews_with_sentiment.groupby('listing_id').agg({
        'comments_negative': ['mean', 'std', 'count'],
        'comments_positive': ['mean', 'std'],
        'comments_neutral': ['mean', 'std'],
        'comments_compound': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    sentiment_agg.columns = ['listing_id'] + [
        f'review_{col[0]}_{col[1]}' for col in sentiment_agg.columns[1:]
    ]
    
    # Merge with listings
    result_df = listings_df.merge(sentiment_agg, on='listing_id', how='left')
    
    # Fill NaN values with 0 for sentiment features
    sentiment_cols = [col for col in result_df.columns if 'review_' in col]
    result_df[sentiment_cols] = result_df[sentiment_cols].fillna(0)
    
    return result_df
