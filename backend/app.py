from flask import Flask, request, jsonify
from flask_cors import CORS  
from flask_pymongo import PyMongo
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import bcrypt  
from pymongo import MongoClient


app = Flask(__name__)
CORS(app) 

app.config["MONGO_URI"] = "mongodb://localhost:27017/ml_project"
mongo = PyMongo(app)



client = MongoClient('mongodb://localhost:27017/')
db = client['ml_project']


customer_data = pd.read_csv('../data/CustomerDataTable.csv')
purchase_history = pd.read_csv('../data/PurchaseHistoryTable.csv')
product_details = pd.read_csv('../data/ProductDetailsTable.csv')


merged_data = pd.merge(purchase_history, product_details, on='Product_ID', how='inner')
merged_data = pd.merge(merged_data, customer_data, on='Customer_ID', how='inner')


ratings_matrix = merged_data.pivot_table(index='Customer_ID', columns='Product_ID', values='Rating')
ratings_matrix_filled = ratings_matrix.fillna(0)


item_similarity = cosine_similarity(ratings_matrix_filled.T)
item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix_filled.columns, columns=ratings_matrix_filled.columns)

def predict_ratings(ratings_matrix, similarity_matrix):
    epsilon = 1e-9
    sum_of_similarities = np.array([np.abs(similarity_matrix).sum(axis=1)]) + epsilon
    ratings_pred = np.dot(ratings_matrix, similarity_matrix) / sum_of_similarities
    ratings_pred = np.nan_to_num(ratings_pred, nan=ratings_matrix.mean().mean())
    return ratings_pred


ratings_pred = predict_ratings(ratings_matrix_filled.values, item_similarity)
ratings_pred_df = pd.DataFrame(ratings_pred, index=ratings_matrix_filled.index, columns=ratings_matrix_filled.columns)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    top_n = request.args.get('top_n')

    
    if not user_id or not top_n:
        return jsonify({"error": "User ID and top_n are required parameters."}), 400

    try:
        user_id = int(user_id)
        top_n = int(top_n)
    except ValueError:
        return jsonify({"error": "User ID and top_n must be integers."}), 400

    
    if user_id not in ratings_matrix_filled.index:
        return jsonify({"error": "User ID not found"}), 404

    user_ratings = ratings_matrix_filled.loc[user_id]
    user_predicted_ratings = ratings_pred_df.loc[user_id]

    
    unrated_items = user_ratings[user_ratings == 0].index

    
    recommended_items = user_predicted_ratings.loc[unrated_items].nlargest(top_n)

    
    recommended_products = product_details[product_details['Product_ID'].isin(recommended_items.index)]

    return jsonify(recommended_products[['Product_ID', 'Product_Name', 'Category']].to_dict(orient='records'))

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    user_id = data.get('userId')
    recommendation_id = data.get('recommendationId')
    feedback = data.get('feedback')

    
    if not user_id or not recommendation_id or feedback is None:
        return jsonify({"error": "userId, recommendationId, and feedback are required."}), 400

    try:
        
        print(f"Feedback from User {user_id} for Product {recommendation_id}: {feedback}")
        return jsonify({"message": "Feedback submitted successfully!"}), 200
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        return jsonify({"error": "Failed to submit feedback."}), 500

@app.route('/signup', methods=['POST'])
def sign_up():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    
    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    
    mongo.db.users.insert_one({
        'email': email,
        'password': hashed_password
    })
    return jsonify({"message": "User registered successfully!"}), 201

@app.route('/signin', methods=['POST'])
def sign_in():
    data = request.json
    email = data.get('email')
    password = data.get('password')

   
    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = mongo.db.users.find_one({'email': email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({"message": "Sign in successful!"}), 200
    else:
        return jsonify({"error": "Invalid email or password."}), 401

if __name__ == '__main__':
    app.run(debug=True)
