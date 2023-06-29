"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_imgorvideo(gr_video, gr_img, text_input, chat_state,chatbot):
    if gr_img is None and gr_video is None:
        return None, None, None, gr.update(interactive=True), chat_state, None
    elif gr_img is not None and gr_video is None:
        print(gr_img)
        chatbot = chatbot + [((gr_img,), None)]
        chat_state = Conversation(
            system= "You are able to understand the visual content that the user provides."
           "Follow the instructions carefully and explain your answers in detail.",
            roles=("Human", "Assistant"),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        )
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    elif gr_video is not None and gr_img is None:
        print(gr_video)
        chatbot = chatbot + [((gr_video,), None)]
        chat_state = default_conversation.copy()
        chat_state = Conversation(
            system= "You are able to understand the visual content that the user provides."
           "Follow the instructions carefully and explain your answers in detail.",
            roles=("Human", "Assistant"),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        )
        img_list = []
        llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    else:
        # img_list = []
        return gr.update(interactive=False), gr.update(interactive=False, placeholder='Currently, only one input is supported'), gr.update(value="Currently, only one input is supported", interactive=False), chat_state, None,chatbot

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    print(chat_state.get_prompt())
    print(chat_state)
    return chatbot, chat_state, img_list

title = """
<h1 align="center"><a href="https://github.com/DAMO-NLP-SG/Video-LLaMA"><img src="https://s1.ax1x.com/2023/05/22/p9oQ0FP.jpg", alt="Video-LLaMA" border="0" style="margin: 0 auto; height: 200px;" /></a> </h1>

<h1 align="center">Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding</h1>

<h5 align="center">  Introduction: Video-LLaMA is a multi-model large language model that achieves video-grounded conversations between humans and computers \
    by connecting language decoder with off-the-shelf unimodal pre-trained models. </h5> 

<div style='display:flex; gap: 0.25rem; '>
<a href='https://github.com/DAMO-NLP-SG/Video-LLaMA'><img src='https://img.shields.io/badge/Github-Code-success'></a>
<a href='https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> 
<a href='https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> 
<a href='https://modelscope.cn/studios/damo/video-llama/summary'><img src='https://img.shields.io/badge/ModelScope-Demo-blueviolet'></a> 
<a href='https://arxiv.org/abs/2306.02858'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>


Thank you for using the Video-LLaMA Demo Page! If you have any questions or feedback, feel free to contact us. 

If you find Video-LLaMA interesting, please give us a star on GitHub.

Current online demo uses the 7B version of Video-LLaMA due to resource limitations. We have released \
         the 13B version on our GitHub repository.


"""

Note_markdown = ("""
### Note
Video-LLaMA is a prototype model and may have limitations in understanding complex scenes, long videos, or specific domains.
The output results may be influenced by input quality, limitations of the dataset, and the model's susceptibility to illusions. Please interpret the results with caution.

**Copyright 2023 Alibaba DAMO Academy.**
""")

cite_markdown = ("""
## Citation
If you find our project useful, hope you can star our repo and cite our paper as follows:
```
@article{damonlpsg2023videollama,
  author = {Zhang, Hang and Li, Xin and Bing, Lidong},
  title = {Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding},
  year = 2023,
  journal = {arXiv preprint arXiv:2306.02858}
  url = {https://arxiv.org/abs/2306.02858}
}
""")

case_note_upload = ("""
### We provide some examples at the bottom of the page. Simply click on them to try them out directly.
""")

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=0.5):
            video = gr.Video()
            image = gr.Image(type="filepath")
            gr.Markdown(case_note_upload)

            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            audio = gr.Checkbox(interactive=True, value=False, label="Audio")
            gr.Markdown(Note_markdown)
        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='Video-LLaMA')
            text_input = gr.Textbox(label='User', placeholder='Upload your image/video first, or directly click the examples at the bottom of the page.', interactive=False)
            

    with gr.Column():
        gr.Examples(examples=[
            [f"examples/dog.jpg", "Which breed is this dog? "],
            [f"examples/JonSnow.jpg", "Who's the man on the right? "],
            [f"examples/Statue_of_Liberty.jpg", "Can you tell me about this building? "],
        ], inputs=[image, text_input])

        gr.Examples(examples=[
            [f"examples/skateboarding_dog.mp4", "What is the dog doing? "],
            [f"examples/birthday.mp4", "What is the boy doing? "],
            [f"examples/IronMan.mp4", "Is the guy in the video Iron Man? "],
        ], inputs=[video, text_input])
        
    gr.Markdown(cite_markdown)
    upload_button.click(upload_imgorvideo, [video, image, text_input, chat_state,chatbot], [video, image, text_input, upload_button, chat_state, img_list,chatbot])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, video, image, text_input, upload_button, chat_state, img_list], queue=False)
    
demo.launch(share=False, enable_queue=True)


# %%
