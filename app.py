import os

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import load_model
from keras.preprocessing import image

import base64
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_player as player

import glob, os
# global variable
fn_count = 0
import os


UPLOAD_DIRECTORY = "/home/avi/Documents/pycharm/xray-project/uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server)


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.layout = html.Div(

    [
        html.H1(" Pneumonia-detection",style={'color':'orange'}),
        html.H2("Upload image"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),   html.H2("File List"),
        html.Ul(id="file-list"),


        # main function place
        html.Div([
            html.Button('detect', id='button_operation',style={
                "backgroundColor":"#4CAF50",
                "color":"black",
                'fontType':'bold',
                "size":'23px',
                "padding":'14px 40px',
                'fontSize':'15px',
                'margin':'20px',
                'marginLeft':'80px'
            }),
            html.Div(id='crack_output',
             children=' CONDITION : '),

            ],
            style={
                'padding': '20px 20px 20px 10px',
                "borderWidth": "2px",
                "borderStyle": "double ",
                "borderRadius": "5px",
            }),

    ],
    style={
        "max-width": "500px",
        "padding" : "2% 20% 2% 25%",
        "margin" : "100px",
        "borderWidth" : "2px",
        "borderRadius" : '2px',
        "borderStyle" : 'dashed',
        "margin":'200px'
    },

)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"),Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]



@app.callback(
    Output("crack_output", "children"),
    [Input('button_operation', 'n_clicks')]
)
def on_click(n_clicks):
    images = glob.glob('uploaded_files/*.*')
    res = ''
    if len(images) != 0:
        loaded_model = tf.keras.models.load_model('res4_4x4_bf_BGR_shu_clahe_red_data.h5')


        image_path = glob.glob('uploaded_files/*.*')[0]
        img = image.load_img(image_path, target_size=(299, 299))
        # plt.imshow(img)
        img = np.expand_dims(img, axis=0)
        result = loaded_model.predict(img).argmax(axis=0)
        r = result[0]
        if int(r) == 0:
            res = 'NORMAL'
        elif int(r) == 1:
            res = 'AFFECTED'

    # REMOVING THE  IMAGES
    imgs = glob.glob('uploaded_files/*.*')
    for img in imgs:
        print(img,'removed...')
        os.remove(img)

    return [html.H1(' condition: '+res)]



if __name__ == "__main__":

    files = os.listdir()
    for x in files:
        if x.startswith('events'):
            os.remove(x);
            print(x)
    I_files = glob.glob('uploaded_files/*.*')
    for x in I_files:
        print(x,'removed...')
        os.remove(x)
    app.run_server(debug=True, port=8888)