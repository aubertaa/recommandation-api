# recommendation.py
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from models import UserData, UserPurchase
from data_loader import load_training_data


def generate_recommendations(user_data: UserData):
    # load training data
    # TODO this csv file has to be defined to train model properly
    training_data = load_training_data("training_data.csv")

    # remove target column from training data
    col_target = "game_id"

    # Columns that are not interesting for the model
    irrelevant_cols = ["game_name", "quantity"]

    # Columns that are categorical : to consider for one hot encoding
    categorical_columns = ["genre", "type", "publisher", "author"]

    # Columns that are numerical : to normalize
    numerical_columns = ["rating", "number_of_reviews", "price"]

    # Separate features (X) and target (y)
    X = training_data.drop(columns=[col_target] + irrelevant_cols)
    y = training_data[col_target]

    # Define preprocessing for categorical and numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ]
    )

    # Create a pipeline with preprocessing and KNN model
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("knn", KNeighborsRegressor(n_neighbors=5)),
        ]
    )

    # Train the model
    model_pipeline.fit(X, y)

    # Predict game IDs for user based on their purchases
    purchased_game_ids = [purchase.game_id for purchase in user_data.purchases]
    purchased_ratings = np.array([purchase.rating for purchase in user_data.purchases])
    # Normalize the ratings to sum to 1
    normalized_ratings = purchased_ratings / np.sum(purchased_ratings)

    # Extract characteristics of purchased games
    purchased_characteristics = training_data[training_data["game_id"].isin(purchased_game_ids)].drop(
        columns=[col_target] + irrelevant_cols
    )

    if purchased_characteristics.size == 0:
        # If user hasn't purchased any games, return nothing
        return []

    # Apply preprocessing to purchased characteristics
    purchased_characteristics_transformed = preprocessor.transform(purchased_characteristics)

    # Calculate the average feature vector for the user's purchased games
    user_preference_vector = np.average(purchased_characteristics_transformed, axis=0, weights=normalized_ratings).reshape(1, -1)

    # Use the KNN model to find similar games
    distances, indices = model_pipeline.named_steps["knn"].kneighbors(user_preference_vector, n_neighbors=5)
    recommended_game_ids = y.iloc[indices[0]]

    # Exclude games already purchased
    recommended_game_ids = [game_id for game_id in recommended_game_ids if game_id not in purchased_game_ids]

    # Map recommendations to game names
    recommendations = [
        {
            "game_id": int(game_id),
            "game_name": training_data[training_data["game_id"] == game_id]["game_name"].values[0]
        }
        for game_id in recommended_game_ids
    ]
    return recommendations


# Example usage
if __name__ == "__main__":
    # Mock user data
    user_data = UserData(
        user_id=1,
        purchases=[
            UserPurchase(game_id=101, rating=4.5),
            UserPurchase(game_id=102, rating=3.0)
        ]
    )

    recs = generate_recommendations(user_data)
    print("Recommendations:", recs)
