from flask import Flask
from app.helpers import initialize_chat_model_factory

def create_app():
    """Create and configure a Flask application instance."""
    app = Flask(__name__)
    app.config.from_object('config.Config')

    # Initialize the chat model without passing individual parameters
    with app.app_context():
        app.chat = initialize_chat_model_factory()

    # Register the API blueprint
    from app.views.api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app

