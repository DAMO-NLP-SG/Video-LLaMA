from flask import Flask
from app.helpers import initialize_chat_model_factory

def create_app():
    """Create and configure a Flask application instance."""
    app = Flask(__name__)
    app.config.from_object('config.Config')

    # Initialize the chat model using the configuration variables from `config.py`
    app.chat = initialize_chat_model_factory(
        config_path=app.config['CONFIG_PATH'],
        model_type=app.config['MODEL_TYPE'],
        device_type=app.config['DEVICE_TYPE']
    )

    # Register the API blueprint
    from app.views.api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app

