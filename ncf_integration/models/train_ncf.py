"""
Neural Collaborative Filtering Training Module
Handles model training, validation, and checkpointing
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

from ncf_integration.models.ncf_model import NCFModel, AdvancedNCF


class NCFTrainer:
    """
    Trainer class for Neural Collaborative Filtering model.
    Handles data preparation, training, and model evaluation.
    """
    
    def __init__(self, model, data_path, batch_size=256, epochs=50, 
                 validation_split=0.2, learning_rate=0.001):
        """
        Initialize NCF trainer.
        
        Args:
            model: NCF model instance
            data_path: Path to the interaction dataset
            batch_size: Batch size for training
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.history = None
        
    def load_data(self):
        """
        Load and preprocess user-food interaction data.
        """
        print("Loading interaction data...")
        interactions_df = pd.read_csv(self.data_path)
        
        # Normalize ratings to [0, 1] range for sigmoid activation
        # Assuming ratings are on 1-5 scale
        interactions_df['rating_normalized'] = (interactions_df['rating'] - 1) / 4
        
        # Shuffle data
        interactions_df = interactions_df.sample(frac=1).reset_index(drop=True)
        
        # Split into train, validation, and test sets
        n = len(interactions_df)
        train_end = int(n * (1 - self.validation_split - 0.1))
        val_end = int(n * (1 - 0.1))
        
        self.train_data = interactions_df[:train_end]
        self.val_data = interactions_df[train_end:val_end]
        self.test_data = interactions_df[val_end:]
        
        print(f"Train samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.val_data)}")
        print(f"Test samples: {len(self.test_data)}")
        
        return self.train_data, self.val_data, self.test_data
    
    def create_dataset(self, data_df, shuffle=True):
        """
        Create TensorFlow dataset from pandas DataFrame.
        
        Args:
            data_df: DataFrame with user_id, food_id, rating columns
            shuffle: Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset object
        """
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'user_input': data_df['user_id'].values,
                'item_input': data_df['food_id'].values
            },
            data_df['rating_normalized'].values
        ))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train(self, model_save_path=None, early_stopping_patience=10):
        """
        Train the NCF model.
        
        Args:
            model_save_path: Path to save the best model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        if self.train_data is None:
            self.load_data()
        
        # Create TensorFlow datasets
        train_dataset = self.create_dataset(self.train_data, shuffle=True)
        val_dataset = self.create_dataset(self.val_data, shuffle=False)
        
        # Callbacks
        callbacks = []
        
        # Model checkpoint
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Train the model
        print(f"\nTraining NCF model for {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        self.history = self.model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining complete!")
        
        return self.history
    
    def evaluate(self, test_data=None):
        """
        Evaluate the trained model on test data.
        
        Args:
            test_data: Test dataset (uses self.test_data if not provided)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if test_data is None:
            test_data = self.test_data
        
        if test_data is None:
            raise ValueError("No test data available. Call load_data() first.")
        
        test_dataset = self.create_dataset(test_data, shuffle=False)
        
        print("\nEvaluating model on test data...")
        results = self.model.model.evaluate(test_dataset, verbose=1)
        
        metrics = {
            'loss': results[0],
            'mae': results[1] if len(results) > 1 else None
        }
        
        print(f"\nTest Loss: {metrics['loss']:.4f}")
        if metrics['mae']:
            print(f"Test MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training and validation loss curves.
        
        Args:
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot (if available)
        if 'mae' in self.history.history:
            axes[1].plot(self.history.history['mae'], label='Train MAE')
            axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
            axes[1].set_title('Model MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_training_config(self, save_path):
        """
        Save training configuration and history.
        
        Args:
            save_path: Path to save the configuration
        """
        config = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'learning_rate': self.learning_rate,
            'train_samples': len(self.train_data) if self.train_data else 0,
            'val_samples': len(self.val_data) if self.val_data else 0,
            'test_samples': len(self.test_data) if self.test_data else 0,
            'training_date': datetime.now().isoformat(),
            'final_train_loss': self.history.history['loss'][-1] if self.history else None,
            'final_val_loss': self.history.history['val_loss'][-1] if self.history else None,
        }
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Training configuration saved to {save_path}")


class NegativeSampling:
    """
    Negative sampling for implicit feedback recommendation.
    Generates negative samples for training.
    """
    
    def __init__(self, num_items, num_negatives=4):
        """
        Initialize negative sampler.
        
        Args:
            num_items: Total number of items
            num_negatives: Number of negative samples per positive sample
        """
        self.num_items = num_items
        self.num_negatives = num_negatives
    
    def generate_negative_samples(self, user_item_pairs):
        """
        Generate negative samples for given user-item pairs.
        
        Args:
            user_item_pairs: DataFrame with user_id and item_id columns
            
        Returns:
            DataFrame with negative samples
        """
        negative_samples = []
        
        for _, row in user_item_pairs.iterrows():
            user_id = row['user_id']
            positive_item = row['food_id']
            
            # Generate negative samples
            for _ in range(self.num_negatives):
                negative_item = np.random.randint(0, self.num_items)
                
                # Ensure negative item is different from positive
                while negative_item == positive_item:
                    negative_item = np.random.randint(0, self.num_items)
                
                negative_samples.append({
                    'user_id': user_id,
                    'food_id': negative_item,
                    'rating': 0  # Negative sample
                })
        
        return pd.DataFrame(negative_samples)


def main():
    """
    Main function to train the NCF model.
    """
    print("=== Neural Collaborative Filtering Training ===\n")
    
    # Load data to get dimensions
    data_path = 'ncf_integration/data/user_food_interactions.csv'
    interactions_df = pd.read_csv(data_path)
    
    num_users = interactions_df['user_id'].nunique()
    num_items = interactions_df['food_id'].nunique()
    
    print(f"Dataset dimensions:")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Interactions: {len(interactions_df)}")
    
    # Create NCF model
    print("\nBuilding NCF model...")
    ncf_model = NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=32,
        hidden_layers=[64, 32, 16],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    model = ncf_model.build_simple_ncf()
    ncf_model.get_model_summary()
    
    # Create trainer
    trainer = NCFTrainer(
        model=ncf_model,
        data_path=data_path,
        batch_size=256,
        epochs=50,
        validation_split=0.2,
        learning_rate=0.001
    )
    
    # Train model
    model_save_path = 'ncf_integration/models/ncf_model.keras'
    history = trainer.train(
        model_save_path=model_save_path,
        early_stopping_patience=10
    )
    
    # Evaluate model
    metrics = trainer.evaluate()
    
    # Plot training history
    trainer.plot_training_history(
        save_path='ncf_integration/models/training_history.png'
    )
    
    # Save training configuration
    trainer.save_training_config(
        save_path='ncf_integration/models/training_config.json'
    )
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: {model_save_path}")
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
