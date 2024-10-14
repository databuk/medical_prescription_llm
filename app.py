from flask import Flask, request, render_template
from transformers import  AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)
model_name = './finetuned_tinybert_drug'
tokenizer_name = './tinybert_tokenizer_drug'

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=99).to(device)
    
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
idx_to_drug = joblib.load('idx_to_drug.joblib')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    condition = request.form.get('condition')
    if not condition:
        return render_template('index.html', error="Please, what's the problem?")
    with torch.no_grad():
        inputs = tokenizer(condition, return_tensors='pt').to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        prediction_id = torch.argmax(logits, dim=1).item()
        prediction = idx_to_drug[prediction_id]
    return render_template('index.html', condition=condition, predicted_drug=prediction)

if __name__ == '__main__':
    app.run(debug=True)
