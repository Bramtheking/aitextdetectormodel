from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("akshayvkt/detect-ai-text")
model = AutoModelForSequenceClassification.from_pretrained("akshayvkt/detect-ai-text")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    text = data.get('text', '')

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).numpy()[0]

    # Return the probabilities for AI and human text
    return jsonify({
        'AI_probability': float(probabilities[1]),
        'Human_probability': float(probabilities[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)