import gradio as gr
from model import get_prediction

def predict(text):
    return get_prediction(text)

gr.Interface(fn=predict, inputs="text", outputs="text").launch()
