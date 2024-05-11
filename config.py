
class Config:
    """General configuration class."""
    # Define paths and model settings
    HUGGING_FACE_HUB_TOKEN = ''
    GPU_ID = 1
    CONFIG_PATH = "eval_configs/video_llama_eval_only_vl.yaml"  # Update this with the actual path to your configuration file
    MODEL_TYPE = 'llama_v2'
    DEVICE_TYPE = 'cpu'
    OPTIONS = {
        "batch_size": 32,
        "learning_rate": 0.01
    }

    DEBUG = True
    SECRET_KEY = "your-secret-key"

