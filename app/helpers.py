import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation, conv_llava_llama_2
import gradio as gr
from flask import current_app

def setup_seeds(seed):
    """Set up random seeds for reproducibility."""
    seed += get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def initialize_chat_model_factory():
    """
    Factory to initialize and return a Chat instance using the current app config.

    Returns:
        Chat: An initialized Chat instance with the appropriate model and processors.
    """
    # Access configuration settings directly from the Flask app config
    cfg_path = current_app.config['CONFIG_PATH']
    options = current_app.config['OPTIONS']
    model_type = current_app.config['MODEL_TYPE']
    device_type = current_app.config['DEVICE_TYPE']

    # Create a mock arguments class to satisfy the `Config` class
    class MockArgs:
        def __init__(self, cfg_path, options):
            self.cfg_path = cfg_path
            self.options = [f"{k}={v}" for k, v in options.items()]

    mock_args = MockArgs(cfg_path, options)

    # Initialize the Config object with the mock arguments
    config = Config(mock_args)
    model_config = config.model_cfg
    setup_seeds(config.run_cfg.seed)

    # Initialize the model based on the specified type
    device = torch.device(device_type)
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    # Set up the visual processor
    vis_processor_cfg = config.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # Return the initialized Chat instance
    return Chat(model, vis_processor, device=device)

def create_demo(chat):
    """Define the Gradio demo using the given chat model."""
    def gradio_reset(chat_state, img_list):
        if chat_state is not None:
            chat_state.messages = []
        if img_list is not None:
            img_list = []
        return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False), gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

    def upload_imgorvideo(gr_video, gr_img, text_input, chat_state, chatbot):
        chat_state = default_conversation.copy() if "vicuna" == 'vicuna' else conv_llava_llama_2.copy()
        if gr_img is None and gr_video is None:
            return None, None, None, gr.update(interactive=True), chat_state, None
        elif gr_img is not None and gr_video is None:
            chatbot = chatbot + [((gr_img,), None)]
            chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            img_list = []
            chat.upload_img(gr_img, chat_state, img_list)
            return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list, chatbot
        elif gr_video is not None and gr_img is None:
            chatbot = chatbot + [((gr_video,), None)]
            chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            img_list = []
            chat.upload_video_without_audio(gr_video, chat_state, img_list)
            return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list, chatbot
        else:
            return gr.update(interactive=False), gr.update(interactive=False, placeholder='Currently, only one input is supported'), gr.update(value="Currently, only one input is supported", interactive=False), chat_state, None, chatbot

    def gradio_ask(user_message, chatbot, chat_state):
        if len(user_message) == 0:
            return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
        chat.ask(user_message, chat_state)
        chatbot = chatbot + [[user_message, None]]
        return '', chatbot, chat_state

    def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
        llm_message = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=300,
            max_length=2000
        )[0]
        chatbot[-1][1] = llm_message
        return chatbot, chat_state, img_list

    title = """
    <h1 align="center">Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding</h1>
    """

    with gr.Blocks() as demo:
        gr.Markdown(title)

        with gr.Row():
            with gr.Column(scale=0.5):
                video = gr.Video()
                image = gr.Image(type="filepath")

                upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
                clear = gr.Button("Restart")
                
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    interactive=True,
                    label="Beam search numbers",
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

            with gr.Column():
                chat_state = gr.State()
                img_list = gr.State()
                chatbot = gr.Chatbot(label='Video-LLaMA')
                text_input = gr.Textbox(label='User', placeholder='Upload your image/video first.')

        upload_button.click(upload_imgorvideo, [video, image, text_input, chat_state, chatbot], [video, image, text_input, upload_button, chat_state, img_list, chatbot])
        
        text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
            gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
        )
        clear.click(gradio_reset, [chat_state, img_list], [chatbot, video, image, text_input, upload_button, chat_state, img_list], queue=False)

    return demo

