from flask import Blueprint, render_template, jsonify, current_app
from app.helpers import create_demo

# Define the blueprint
api_bp = Blueprint('api', __name__)

# Register the blueprint route for the index page
@api_bp.route('/')
def index():
    return render_template('index.html')

@api_bp.route('/demo')
def run_demo():
    # Access the pre-initialized chat model via `current_app`
    chat = current_app.chat
    demo = create_demo(chat)
    demo.launch(share=True, enable_queue=True)
    return jsonify(message="Gradio demo is running.")

