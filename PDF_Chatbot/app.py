# app.py
import os
import torch
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, AutoModelForQuestionAnswering
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'C:\\Users\\middh\\PDF_Chatbot'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def read_pdf(file_path):
    with fitz.open(file_path) as pdf_doc:
        text = ''
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            text += page.get_text()
        return text

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html', answer=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_file' not in request.files:
        return 'No file part'
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        filename = secure_filename('uploaded_pdf.pdf')
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify(success=True, message='File uploaded successfully')
    else:
        return jsonify(success=False, message='Invalid file format. Only PDF files are allowed.')

@app.route('/chat', methods=['POST'])
def chat():
    if 'question' in request.form:
        question = request.form['question']
        pdf_text = read_pdf(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_pdf.pdf'))

        inputs = tokenizer.encode_plus(question, pdf_text, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
        start_scores, end_scores = model(**inputs).start_logits, model(**inputs).end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        
        answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1])
        return render_template('index.html', answer=answer)
    else:
        return jsonify(success=False, message='Invalid request')

if __name__ == '__main__':
    app.run(debug=True)
