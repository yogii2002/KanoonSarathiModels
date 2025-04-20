import gradio as gr
from model import prediction

def predict(text):
    return prediction(text)

gr.Interface(fn=predict, inputs="text", outputs="text").launch()
