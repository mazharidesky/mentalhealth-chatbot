from flask import Flask, render_template, request, jsonify
from chatbot import ContextualMentalHealthChatbot
import os


data = [
    {
        "image": "../static/img/psikiater-1.png",
        "name": "Arman Sp.Kj",
        "specialist": "Konsultan Kesehatan Jiwa",
        "hubungi": "08123456789"
    },
    
    {
        "image": "../static/img/psikiater-1.png",
        "name": "Arman Sp.Kj",
        "specialist": "Konsultan Kesehatan Jiwa",
        "hubungi": "08123456789"
    },
    
    {
        "image": "../static/img/psikiater-1.png",
        "name": "Arman Sp.Kj",
        "specialist": "Konsultan Kesehatan Jiwa",
        "hubungi": "08123456789"
    },
    
    {
        "image": "../static/img/psikiater-1.png",
        "name": "Arman Sp.Kj",
        "specialist": "Konsultan Kesehatan Jiwa",
        "hubungi": "08123456789"
    },
    
]

app = Flask(__name__, static_folder='static')
chatbot = ContextualMentalHealthChatbot('data.json')

@app.route('/')
def home():
    return render_template('index.html', data=data)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = chatbot.get_response(user_message)
    return jsonify({
        'response': response['response'],
        'tag': response['tag'],
        'confidence': response['confidence']
    })

if __name__ == '__main__':
    app.run(debug=True)