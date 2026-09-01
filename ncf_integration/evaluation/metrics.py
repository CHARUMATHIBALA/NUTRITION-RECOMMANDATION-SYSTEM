"""
Evaluation Metrics for Neural Collaborative Filtering
Implements standard recommendation system evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import math


class RecommendationMetrics:
    """
    Comprehensive evaluation metrics for recommendation systems.
    Includes ranking-based and rating-based metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_history = {}
    
    @staticmethod
    def precision_at_k(relevant_items: List[int], recommended_items: List[int], k: int) -> float:
        """
        Calculate Precision@K.
        
        Precision@K measures the fraction of recommended items that are relevant.
        
        Mathematical Formula:
        Precision@K = |{relevant items} ∩ {top-K recommended}| / K
        
        Args:
            relevant_items: List of relevant item IDs for the user
            recommended_items: List of recommended item IDs (sorted by relevance)
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score (0 to 1)
        """
        if k == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(relevant_items: List[int], recommended_items: List[int], k: int) -> float:
        """
        Calculate Recall@K.
        
        Recall@K measures the fraction of relevant items that appear in top-K recommendations.
        
        Mathematical Formula:
        Recall@K = |{relevant items} ∩ {top-K recommended}| / |{relevant items}|
        
        Args:
            relevant_items: List of relevant item IDs for the user
            recommended_items: List of recommended item IDs (sorted by relevance)
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score (0 to 1)
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        
        return relevant_in_top_k / len(relevant_items)
    
    @staticmethod
    def f1_at_k(relevant_items: List[int], recommended_items: List[int], k: int) -> float:
        """
        Calculate F1@K (harmonic mean of Precision@K and Recall@K).
        
        Mathematical Formula:
        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
        
        Args:
            relevant_items: List of relevant item IDs for the user
            recommended_items: List of recommended item IDs (sorted by relevance)
            k: Number of top recommendations to consider
            
        Returns:
            F1@K score (0 to 1)
        """
        precision = RecommendationMetrics.precision_at_k(relevant_items, recommended_items, k)
        recall = RecommendationMetrics.recall_at_k(relevant_items, recommended_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(relevant_items: List[int], recommended_items: List[int], 
                  relevance_scores: Dict[int, float] = None, k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        NDCG measures ranking quality by considering the position of relevant items.
        Higher positions are discounted logarithmically.
        
        Mathematical Formula:
        DCG@K = Σ (2^rel_i - 1) / log2(i + 1)
        NDCG@K = DCG@K / IDCG@K
        
        where rel_i is the relevance of item at position i,
        and IDCG is the ideal DCG (perfect ranking).
        
        Args:
            relevant_items: List of relevant item IDs for the user
            recommended_items: List of recommended item IDs (sorted by relevance)
            relevance_scores: Dictionary mapping item IDs to relevance scores (optional)
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score (0 to 1)
        """
        if relevance_scores is None:
            # Binary relevance: relevant items have score 1, others 0
            relevance_scores = {item: 1.0 for item in relevant_items}
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            rel = relevance_scores.get(item, 0.0)
            dcg += (2**rel - 1) / math.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = sorted([relevance_scores.get(item, 0.0) 
                                   for item in relevant_items], reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances[:k]):
            idcg += (2**rel - 1) / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def hit_ratio_at_k(relevant_items: List[int], recommended_items: List[int], k: int) -> float:
        """
        Calculate Hit Ratio@K.
        
        Hit Ratio@K measures whether at least one relevant item appears in top-K.
        
        Mathematical Formula:
        Hit@K = 1 if |{relevant items} ∩ {top-K recommended}| > 0 else 0
        
        Args:
            relevant_items: List of relevant item IDs for the user
            recommended_items: List of recommended item IDs (sorted by relevance)
            k: Number of top recommendations to consider
            
        Returns:
            Hit Ratio@K score (0 or 1)
        """
        top_k = recommended_items[:k]
        return 1.0 if len(set(top_k) & set(relevant_items)) > 0 else 0.0
    
    @staticmethod
    def mrr(relevant_items: List[int], recommended_items: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR measures the rank of the first relevant item in the recommendation list.
        
        Mathematical Formula:
        MRR = 1 / rank_of_first_relevant_item
        
        Args:
            relevant_items: List of relevant item IDs for the user
            recommended_items: List of recommended item IDs (sorted by relevance)
            
        Returns:
            MRR score (0 to 1)
        """
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error (RMSE).
        
        RMSE measures the average magnitude of prediction errors.
        
        Mathematical Formula:
        RMSE = sqrt(Σ (pred_i - actual_i)^2 / n)
        
        Args:
            predictions: Array of predicted ratings
            actuals: Array of actual ratings
            
        Returns:
            RMSE score
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return 0.0
        
        mse = np.mean((predictions - actuals) ** 2)
        return math.sqrt(mse)
    
    @staticmethod
    def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        
        MAE measures the average absolute difference between predictions and actuals.
        
        Mathematical Formula:
        MAE = Σ |pred_i - actual_i| / n
        
        Args:
            predictions: Array of predicted ratings
            actuals: Array of actual ratings
            
        Returns:
            MAE score
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return 0.0
        
        return np.mean(np.abs(predictions - actuals))
    
    @staticmethod
    def coverage(all_items: List[int], recommended_items_per_user: List[List[int]]) -> float:
        """
        Calculate Catalog Coverage.
        
        Coverage measures the fraction of items that appear in recommendations.
        
        Mathematical Formula:
        Coverage = |∪_{u} recommended_u| / |all_items|
        
        Args:
            all_items: List of all item IDs in the catalog
            recommended_items_per_user: List of recommended items for each user
            
        Returns:
            Coverage score (0 to 1)
        """
        if len(all_items) == 0:
            return 0.0
        
        all_recommended = set()
        for recommendations in recommended_items_per_user:
            all_recommended.update(recommendations)
        
        return len(all_recommended) / len(all_items)
    
    @staticmethod
    def diversity(recommended_items: List[int], item_features: pd.DataFrame) -> float:
        """
        Calculate Intra-list Diversity.
        
        Diversity measures how different the recommended items are from each other.
        
        Mathematical Formula:
        Diversity = (1 / (K*(K-1))) * Σ_{i≠j} (1 - similarity(item_i, item_j))
        
        Args:
            recommended_items: List of recommended item IDs
            item_features: DataFrame with item features for similarity calculation
            
        Returns:
            Diversity score (0 to 1)
        """
        if len(recommended_items) < 2:
            return 0.0
        
        # Simple diversity based on item categories
        # In a real implementation, use cosine similarity on feature vectors
        categories = item_features.loc[item_features['food_id'].isin(recommended_items)]['MealType'].values
        unique_categories = len(set(categories))
        
        return unique_categories / len(categories)
    
    def evaluate_user(self, user_id: int, relevant_items: List[int], 
                     recommended_items: List[int], k_values: List[int] = [5, 10, 20],
                     relevance_scores: Dict[int, float] = None) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user.
        
        Args:
            user_id: User ID
            relevant_items: List of relevant items for the user
            recommended_items: List of recommended items
            k_values: List of K values for ranking metrics
            relevance_scores: Optional relevance scores for NDCG
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {'user_id': user_id}
        
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(relevant_items, recommended_items, k)
            metrics[f'recall@{k}'] = self.recall_at_k(relevant_items, recommended_items, k)
            metrics[f'f1@{k}'] = self.f1_at_k(relevant_items, recommended_items, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(relevant_items, recommended_items, 
                                                   relevance_scores, k)
            metrics[f'hit_ratio@{k}'] = self.hit_ratio_at_k(relevant_items, recommended_items, k)
        
        metrics['mrr'] = self.mrr(relevant_items, recommended_items)
        
        return metrics
    
    def evaluate_dataset(self, test_data: pd.DataFrame, model, 
                        k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate recommendations on entire test dataset.
        
        Args:
            test_data: DataFrame with user_id, food_id, rating columns
            model: Trained NCF model
            k_values: List of K values for ranking metrics
            
        Returns:
            Dictionary of average metric scores
        """
        all_metrics = []
        all_items = test_data['food_id'].unique().tolist()
        
        # Group by user
        for user_id, user_data in test_data.groupby('user_id'):
            # Get relevant items (high-rated items)
            relevant_items = user_data[user_data['rating'] >= 4]['food_id'].tolist()
            
            if len(relevant_items) == 0:
                continue
            
            # Get candidate items (all items except user's interacted items)
            user_items = user_data['food_id'].tolist()
            candidate_items = [item for item in all_items if item not in user_items]
            
            # Get predictions
            predictions = model.predict(user_id, candidate_items)
            
            # Sort by predicted rating
            item_predictions = list(zip(candidate_items, predictions))
            item_predictions.sort(key=lambda x: x[1], reverse=True)
            recommended_items = [item for item, _ in item_predictions]
            
            # Evaluate
            user_metrics = self.evaluate_user(
                user_id=user_id,
                relevant_items=relevant_items,
                recommended_items=recommended_items,
                k_values=k_values
            )
            all_metrics.append(user_metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            for col in metrics_df.columns:
                if col != 'user_id':
                    avg_metrics[col] = metrics_df[col].mean()
        
        return avg_metrics
    
    def evaluate_rating_prediction(self, test_data: pd.DataFrame, model) -> Dict[str, float]:
        """
        Evaluate rating prediction accuracy.
        
        Args:
            test_data: DataFrame with user_id, food_id, rating columns
            model: Trained NCF model
            
        Returns:
            Dictionary of rating metrics (RMSE, MAE)
        """
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            food_id = row['food_id']
            actual_rating = row['rating']
            
            predicted_rating = model.predict(user_id, [food_id])[0]
            
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return {
            'rmse': self.rmse(predictions, actuals),
            'mae': self.mae(predictions, actuals)
        }
    
    def print_metrics_report(self, metrics: Dict[str, float]):
        """
        Print a formatted metrics report.
        
        Args:
            metrics: Dictionary of metric scores
        """
        print("\n" + "="*50)
        print("RECOMMENDATION SYSTEM EVALUATION REPORT")
        print("="*50)
        
        # Group metrics by type
        ranking_metrics = {}
        rating_metrics = {}
        
        for key, value in metrics.items():
            if key in ['rmse', 'mae']:
                rating_metrics[key] = value
            else:
                ranking_metrics[key] = value
        
        # Print ranking metrics
        if ranking_metrics:
            print("\n--- Ranking Metrics ---")
            for key, value in sorted(ranking_metrics.items()):
                print(f"{key.upper():20s}: {value:.4f}")
        
        # Print rating metrics
        if rating_metrics:
            print("\n--- Rating Prediction Metrics ---")
            for key, value in sorted(rating_metrics.items()):
                print(f"{key.upper():20s}: {value:.4f}")
        
        print("="*50 + "\n")


def main():
    """
    Test the evaluation metrics implementation.
    """
    print("=== Testing Recommendation Metrics ===\n")
    
    # Create sample data
    relevant_items = [1, 5, 10, 15, 20]
    recommended_items = [5, 3, 10, 8, 15, 2, 20, 1, 7, 12]
    
    # Calculate metrics
    metrics_calc = RecommendationMetrics()
    
    print("Sample Evaluation:")
    print(f"Relevant items: {relevant_items}")
    print(f"Recommended items: {recommended_items[:10]}")
    
    for k in [5, 10]:
        precision = metrics_calc.precision_at_k(relevant_items, recommended_items, k)
        recall = metrics_calc.recall_at_k(relevant_items, recommended_items, k)
        f1 = metrics_calc.f1_at_k(relevant_items, recommended_items, k)
        ndcg = metrics_calc.ndcg_at_k(relevant_items, recommended_items, k=k)
        hit_ratio = metrics_calc.hit_ratio_at_k(relevant_items, recommended_items, k)
        
        print(f"\n@K={k}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  NDCG: {ndcg:.4f}")
        print(f"  Hit Ratio: {hit_ratio:.4f}")
    
    mrr = metrics_calc.mrr(relevant_items, recommended_items)
    print(f"\nMRR: {mrr:.4f}")
    
    # Test rating metrics
    predictions = np.array([4.2, 3.8, 4.5, 2.1, 3.9])
    actuals = np.array([4.0, 4.0, 5.0, 2.0, 4.0])
    
    rmse = metrics_calc.rmse(predictions, actuals)
    mae = metrics_calc.mae(predictions, actuals)
    
    print(f"\nRating Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    print("\nMetrics test complete!")


if __name__ == "__main__":
    main()
