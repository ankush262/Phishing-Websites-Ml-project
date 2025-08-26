import pickle
import numpy as np

model_filename = 'phishing_svm_model.pkl'
try:
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"Model '{model_filename}' loaded successfully. ✅")
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_filename}'.")
    print("Please run the training script first to create the model file.")
    exit()

# --- 2. Prediction Function ---
# This function takes a list or NumPy array of 30 features and returns the prediction.
def predict_website(features):
    """
    Predicts if a website is phishing or legitimate based on its features.

    Args:
        features (list or np.array): A list containing 9 feature values (-1, 0, or 1).

    Returns:
        str: A string indicating the prediction ('Phishing' or 'Legitimate').
    """
    # The model expects a 2D array, so we reshape the input.
    features_array = np.array(features).reshape(1, -1)
    
    # Use the loaded model to make a prediction.
    prediction = loaded_model.predict(features_array)
    
    # Interpret the prediction (0 = Phishing, 1 = Legitimate).
    if prediction[0] == 0:
        return "Prediction: This website is likely PHISHING.  danger" 
    else:
        return "Prediction: This website is likely LEGITIMATE. safe" 


if __name__ == "__main__":
    # Example 1: A sample representing a likely phishing site.
    # (These are hypothetical values for demonstration).
    phishing_example = [-1, 1, 1, 1, -1, -1, -1, -1, -1]
    
    # Example 2: A sample representing a likely legitimate site.
    legitimate_example = [1, 1, 1, 1, 1, -1, 0, 1, 1]
    
    print("\n--- Testing with a likely phishing example ---")
    result1 = predict_website(phishing_example)
    print(result1)

    print("\n--- Testing with a likely legitimate example ---")
    result2 = predict_website(legitimate_example)
    print(result2)
