from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)



API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response.json())
    return response.json()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data['text']
    result = query({"inputs": text})
    
    # Check if the model is still loading
    if 'error' in result and 'loading' in result['error']:
        return jsonify({"message": "Model is currently loading, please try again later."}), 503
    
    # Handle unexpected response structure
    if not isinstance(result, list) or not result:
        return jsonify({"message": "Unexpected response from the model."}), 500
    
    try:
        # Extract the emotion with the highest score
        emotions = result[0]  # Assuming result is a list with one element
        highest_emotion = max(emotions, key=lambda x: x['score'])
        emotion = highest_emotion['label']
    except (KeyError, TypeError, IndexError) as e:
        # Handle any errors during processing
        return jsonify({"message": f"Error processing response: {str(e)}"}), 500
    
    return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(debug=True)
