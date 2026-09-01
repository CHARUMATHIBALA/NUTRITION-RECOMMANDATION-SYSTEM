"""
Neural Collaborative Filtering (NCF) Model Implementation
Based on the NCF architecture from He et al. (2017)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, backend as K
import numpy as np


class NCFModel:
    """
    Neural Collaborative Filtering Model for food recommendation.
    
    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
    to learn non-linear user-item interactions.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=32, 
                 hidden_layers=[64, 32, 16], dropout_rate=0.2,
                 learning_rate=0.001):
        """
        Initialize NCF model.
        
        Args:
            num_users: Number of unique users in the dataset
            num_items: Number of unique food items
            embedding_dim: Dimension of user and item embeddings
            hidden_layers: List of hidden layer sizes for MLP
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.gmf_model = None
        self.mlp_model = None
        
    def build_gmf(self):
        """
        Build Generalized Matrix Factorization (GMF) component.
        GMF learns linear interactions between user and item embeddings.
        
        Mathematical Model:
        h_GMF = p_u ⊙ q_i
        where p_u is user embedding, q_i is item embedding, ⊙ is element-wise product
        """
        # User input and embedding
        user_input = layers.Input(shape=(1,), name='user_input_gmf')
        user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name='user_embedding_gmf'
        )(user_input)
        user_embedding = layers.Flatten()(user_embedding)
        
        # Item input and embedding
        item_input = layers.Input(shape=(1,), name='item_input_gmf')
        item_embedding = layers.Embedding(
            input_dim=self.num_items,
            output_dim=self.embedding_dim,
            name='item_embedding_gmf'
        )(item_input)
        item_embedding = layers.Flatten()(item_embedding)
        
        # Element-wise product of embeddings
        gmf_vector = layers.multiply([user_embedding, item_embedding])
        
        # Output layer
        gmf_output = layers.Dense(1, activation='sigmoid', name='gmf_output')(gmf_vector)
        
        # Create GMF model
        self.gmf_model = Model(
            inputs=[user_input, item_input],
            outputs=gmf_output,
            name='GMF'
        )
        
        return self.gmf_model
    
    def build_mlp(self):
        """
        Build Multi-Layer Perceptron (MLP) component.
        MLP learns non-linear interactions through deep neural networks.
        
        Mathematical Model:
        h_MLP = ReLU(W_L(ReLU(...(W_2(ReLU(W_1[p_u; q_i]) + b_1))...) + b_L))
        where [p_u; q_i] is concatenation of user and item embeddings
        """
        # User input and embedding
        user_input = layers.Input(shape=(1,), name='user_input_mlp')
        user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name='user_embedding_mlp'
        )(user_input)
        user_embedding = layers.Flatten()(user_embedding)
        
        # Item input and embedding
        item_input = layers.Input(shape=(1,), name='item_input_mlp')
        item_embedding = layers.Embedding(
            input_dim=self.num_items,
            output_dim=self.embedding_dim,
            name='item_embedding_mlp'
        )(item_input)
        item_embedding = layers.Flatten()(item_embedding)
        
        # Concatenate embeddings
        mlp_vector = layers.concatenate([user_embedding, item_embedding])
        
        # Hidden layers with ReLU activation and dropout
        for i, hidden_size in enumerate(self.hidden_layers):
            mlp_vector = layers.Dense(
                hidden_size,
                activation='relu',
                name=f'mlp_hidden_{i}'
            )(mlp_vector)
            mlp_vector = layers.Dropout(self.dropout_rate)(mlp_vector)
        
        # Output layer
        mlp_output = layers.Dense(1, activation='sigmoid', name='mlp_output')(mlp_vector)
        
        # Create MLP model
        self.mlp_model = Model(
            inputs=[user_input, item_input],
            outputs=mlp_output,
            name='MLP'
        )
        
        return self.mlp_model
    
    def build_ncf(self, alpha=0.5):
        """
        Build complete NCF model by combining GMF and MLP.
        
        Mathematical Model:
        ŷ_ui = α * h_GMF^T * h + (1-α) * h_MLP^T * h
        where α is the weighting factor between GMF and MLP
        
        Args:
            alpha: Weighting factor for GMF vs MLP (default: 0.5)
        """
        # Build GMF and MLP components
        gmf = self.build_gmf()
        mlp = self.build_mlp()
        
        # Get the embedding layers from both models
        gmf_user_emb = gmf.get_layer('user_embedding_gmf').output
        gmf_item_emb = gmf.get_layer('item_embedding_gmf').output
        mlp_user_emb = mlp.get_layer('user_embedding_mlp').output
        mlp_item_emb = mlp.get_layer('item_embedding_mlp').output
        
        # Flatten embeddings
        gmf_user_emb = layers.Flatten()(gmf_user_emb)
        gmf_item_emb = layers.Flatten()(gmf_item_emb)
        mlp_user_emb = layers.Flatten()(mlp_user_emb)
        mlp_item_emb = layers.Flatten()(mlp_item_emb)
        
        # GMF path: element-wise product
        gmf_vector = layers.multiply([gmf_user_emb, glp_item_emb])
        
        # MLP path: concatenation through hidden layers
        mlp_vector = layers.concatenate([mlp_user_emb, mlp_item_emb])
        for i, hidden_size in enumerate(self.hidden_layers):
            mlp_vector = layers.Dense(hidden_size, activation='relu')(mlp_vector)
            mlp_vector = layers.Dropout(self.dropout_rate)(mlp_vector)
        
        # Combine GMF and MLP outputs
        combined_vector = layers.concatenate([gmf_vector, mlp_vector])
        
        # Final prediction layer
        output = layers.Dense(1, activation='sigmoid', name='ncf_output')(combined_vector)
        
        # Create complete NCF model
        self.model = Model(
            inputs=[gmf.input[0], gmf.input[1]],
            outputs=output,
            name='NCF'
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_simple_ncf(self):
        """
        Build a simplified NCF model for faster training.
        Uses a single neural network with concatenated embeddings.
        """
        # User input and embedding
        user_input = layers.Input(shape=(1,), name='user_input')
        user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)
        user_embedding = layers.Flatten()(user_embedding)
        
        # Item input and embedding
        item_input = layers.Input(shape=(1,), name='item_input')
        item_embedding = layers.Embedding(
            input_dim=self.num_items,
            output_dim=self.embedding_dim,
            name='item_embedding'
        )(item_input)
        item_embedding = layers.Flatten()(item_embedding)
        
        # Concatenate embeddings
        concat_vector = layers.concatenate([user_embedding, item_embedding])
        
        # Hidden layers
        for i, hidden_size in enumerate(self.hidden_layers):
            concat_vector = layers.Dense(
                hidden_size,
                activation='relu',
                name=f'hidden_{i}'
            )(concat_vector)
            concat_vector = layers.Dropout(self.dropout_rate)(concat_vector)
        
        # Output layer (regression for rating prediction)
        output = layers.Dense(1, activation='linear', name='rating_output')(concat_vector)
        
        # Create model
        self.model = Model(
            inputs=[user_input, item_input],
            outputs=output,
            name='SimpleNCF'
        )
        
        # Compile model with MSE loss for rating prediction
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def get_user_embedding(self, user_id):
        """
        Extract learned user embedding for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            User embedding vector
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_ncf() or build_simple_ncf() first.")
        
        embedding_layer = self.model.get_layer('user_embedding')
        embedding_matrix = embedding_layer.get_weights()[0]
        return embedding_matrix[user_id]
    
    def get_item_embedding(self, item_id):
        """
        Extract learned item embedding for a specific food item.
        
        Args:
            item_id: ID of the food item
            
        Returns:
            Item embedding vector
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_ncf() or build_simple_ncf() first.")
        
        embedding_layer = self.model.get_layer('item_embedding')
        embedding_matrix = embedding_layer.get_weights()[0]
        return embedding_matrix[item_id]
    
    def predict(self, user_id, item_ids):
        """
        Predict ratings for a user and multiple items.
        
        Args:
            user_id: ID of the user
            item_ids: List of item IDs to predict ratings for
            
        Returns:
            Array of predicted ratings
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_ncf() or build_simple_ncf() first.")
        
        # Create input arrays
        user_array = np.full(len(item_ids), user_id)
        item_array = np.array(item_ids)
        
        # Make predictions
        predictions = self.model.predict([user_array, item_array], verbose=0)
        
        return predictions.flatten()
    
    def recommend_top_n(self, user_id, item_ids, n=10):
        """
        Get top-N recommendations for a user.
        
        Args:
            user_id: ID of the user
            item_ids: List of candidate item IDs
            n: Number of recommendations to return
            
        Returns:
            List of (item_id, predicted_rating) tuples sorted by rating
        """
        predictions = self.predict(user_id, item_ids)
        
        # Sort by predicted rating
        item_ratings = list(zip(item_ids, predictions))
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        
        return item_ratings[:n]
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Update dimensions from loaded model
        self.num_users = self.model.get_layer('user_embedding').input_dim
        self.num_items = self.model.get_layer('item_embedding').input_dim
    
    def get_model_summary(self):
        """
        Print model architecture summary.
        """
        if self.model is None:
            print("Model not built yet.")
        else:
            self.model.summary()


class AdvancedNCF(NCFModel):
    """
    Advanced NCF model with additional features:
    - Attention mechanism
    - Side information integration
    - Multi-task learning
    """
    
    def __init__(self, num_users, num_items, embedding_dim=32,
                 hidden_layers=[64, 32, 16], dropout_rate=0.2,
                 learning_rate=0.001, use_attention=False):
        """
        Initialize Advanced NCF model.
        
        Args:
            use_attention: Whether to use attention mechanism
        """
        super().__init__(num_users, num_items, embedding_dim, 
                        hidden_layers, dropout_rate, learning_rate)
        self.use_attention = use_attention
    
    def build_with_attention(self):
        """
        Build NCF model with attention mechanism.
        Attention helps the model focus on important features.
        """
        # User and item inputs
        user_input = layers.Input(shape=(1,), name='user_input')
        item_input = layers.Input(shape=(1,), name='item_input')
        
        # Embeddings
        user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)
        user_embedding = layers.Flatten()(user_embedding)
        
        item_embedding = layers.Embedding(
            input_dim=self.num_items,
            output_dim=self.embedding_dim,
            name='item_embedding'
        )(item_input)
        item_embedding = layers.Flatten()(item_embedding)
        
        # Concatenate
        concat = layers.concatenate([user_embedding, item_embedding])
        
        # Attention mechanism
        attention_weights = layers.Dense(
            self.embedding_dim * 2,
            activation='tanh',
            name='attention_weights'
        )(concat)
        attention_weights = layers.Dense(1, activation='softmax')(attention_weights)
        
        # Apply attention
        attended = layers.multiply([concat, attention_weights])
        
        # Hidden layers
        for i, hidden_size in enumerate(self.hidden_layers):
            attended = layers.Dense(hidden_size, activation='relu')(attended)
            attended = layers.Dropout(self.dropout_rate)(attended)
        
        # Output
        output = layers.Dense(1, activation='linear')(attended)
        
        # Create model
        self.model = Model(
            inputs=[user_input, item_input],
            outputs=output,
            name='AttentionNCF'
        )
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model


def main():
    """
    Test the NCF model implementation.
    """
    print("=== Testing NCF Model Implementation ===\n")
    
    # Create a simple test model
    ncf = NCFModel(
        num_users=100,
        num_items=50,
        embedding_dim=16,
        hidden_layers=[32, 16],
        dropout_rate=0.2
    )
    
    # Build simple NCF model
    print("Building Simple NCF Model...")
    model = ncf.build_simple_ncf()
    
    # Print model summary
    print("\nModel Architecture:")
    ncf.get_model_summary()
    
    # Test prediction
    print("\nTesting prediction...")
    test_user = 5
    test_items = [10, 20, 30, 40]
    predictions = ncf.predict(test_user, test_items)
    print(f"Predictions for user {test_user}:")
    for item, pred in zip(test_items, predictions):
        print(f"  Item {item}: {pred:.4f}")
    
    # Test top-N recommendations
    print("\nTesting Top-N recommendations...")
    all_items = list(range(50))
    top_n = ncf.recommend_top_n(test_user, all_items, n=5)
    print(f"Top 5 recommendations for user {test_user}:")
    for item, rating in top_n:
        print(f"  Item {item}: {rating:.4f}")
    
    print("\nNCF Model test complete!")


if __name__ == "__main__":
    main()
