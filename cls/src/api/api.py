from flask import Flask, request, jsonify
from src.models.bert.model_bert import BertModel
from src.models.cnn.model_cnn import CnnModel

app = Flask(__name__)

# Initialize models
bert_model = BertModel()
cnn_model = CnnModel()

@app.route('/predict/bert', methods=['POST'])
def predict_bert():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    prediction = bert_model.predict(text)
    return jsonify({'prediction': prediction})

@app.route('/predict/cnn', methods=['POST'])
def predict_cnn():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    prediction = cnn_model.predict(text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)