from flask import Flask, render_template, request
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

app = Flask(__name__)

# Load the travel ratings data
travel_ratings = pd.read_csv('travel_ratings.csv')

# Define the rating scale
reader = Reader(rating_scale=(1, 5))

# Load the data into the Surprise Dataset format
data = Dataset.load_from_df(travel_ratings[['user_id', 'travel_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Train the SVD algorithm on the training set
algo = SVD()
algo.fit(trainset)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the user ID from the form
    user_id = int(request.form['user_id'])

    # Get the travel IDs that the user has not rated
    rated_travel_ids = travel_ratings[travel_ratings['user_id'] == user_id]['travel_id'].tolist()
    unrated_travel_ids = travel_ratings[~travel_ratings['travel_id'].isin(rated_travel_ids)]['travel_id'].tolist()

    # Predict the ratings for the unrated travel IDs
    predictions = []
    for travel_id in unrated_travel_ids:
        prediction = algo.predict(user_id, travel_id)
        predictions.append((travel_id, prediction.est))

    # Sort the predictions by rating (highest first)
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get the top 5 travel recommendations
    top_recommendations = predictions[:5]

    # Get the travel details for the top recommendations
    travel_details = []
    for travel_id, rating in top_recommendations:
        travel_detail = travel_ratings[travel_ratings['travel_id'] == travel_id].iloc[0]
        travel_details.append((travel_detail['travel_name'], travel_detail['travel_description'], rating))

    # Render the recommendations template with the travel details
    return render_template('recommendations.html', travel_details=travel_details)

if __name__ == '__main__':
    app.run(debug=True)
