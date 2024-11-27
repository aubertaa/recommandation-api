# recommendation.py
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from models import UserData, UserPurchase
from data_loader import load_training_data


def generate_recommendations(user_data: UserData):
    # À compléter avec un vrai algorithme de machine learning

    # Pour l'instant, retourne une liste de jeux en exemple
    # TODO remove this hard coded recommandations
    recommendations = [
        {"game_id": 101, "game_name": "Pandemic"},
        {"game_id": 102, "game_name": "Catan"},
        {"game_id": 103, "game_name": "Ticket to Ride"}
    ]

    #load training data
    #TODO this csv file has to be defined to train model
    training_data = load_training_data("training_data.csv")

    # train KNN model on loaded training data
    X = training_data.iloc[:, 1:-1].values  # Extract game characteristics columns from csv
    y = training_data["game_id"].values     # Game IDs are target labels

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X, y)

    # Predict game IDs for user based on their purchases
    purchased_game_ids = [purchase.game_id for purchase in user_data.purchases]
    purchased_ratings = [purchase.rating for purchase in user_data.purchases]

    # Extract features for purchased games
    purchased_features = training_data[training_data["game_id"].isin(purchased_game_ids)].iloc[:, 1:-1].values

    if purchased_features.size == 0:
        # If user hasn't purchased any games, return nothing
        return []

    # Compute the average preference vector
    user_preference_vector = np.mean(purchased_features, axis=0)

    # Predict similar games
    distances, indices = model.kneighbors([user_preference_vector], return_distance=True)
    recommended_game_ids = y[indices[0]]

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
