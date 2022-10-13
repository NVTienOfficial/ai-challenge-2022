import os
from io import BytesIO
from pyexpat import model
from random import sample
from utils import server_base_path, video_path, keyframe_path, object_path
from multiprocessing import Process
import time
import json

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import clip
import torch
import torch.nn.functional as F 

from flask import Flask, flash, send_file, url_for, Response, send_from_directory
from flask import render_template, request, redirect
from werkzeug.utils import secure_filename

if torch.cuda.is_available():
    device = torch.device("cuda")
    data_type = torch.float16
else:
    device = torch.device("cpu")
    data_type = torch.float32

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = server_base_path / 'static/uploads/'
app.secret_key = "11022022"

@app.before_first_request
def _load_assets():
    app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True, parents=True)
    p = Process(target=delete_uploaded_images)
    p.start()

    # init videos
    global result_indices
    result_indices = []

    # init captions
    global captions
    captions = []

    # load dataset features
    load_video_features()

    # init model
    global clip_model
    global clip_preprocessor
    clip_model, clip_preprocessor = clip.load("ViT-B/16")
    clip_model = clip_model.eval().to(device)

def load_video_features(dim:int =512):
    global video_features
    global video_ids
    global frame_ids
    video_features = torch.empty((0,dim), dtype=data_type).to(device)
    video_ids = []
    frame_ids = []

    for f in os.listdir('./CLIP_features'):
        video_id = f.replace('.npy', '')
        frame_id = []
        for fr in os.listdir('./keyframes/'+video_id):
            frame_id.append(fr.replace('.jpg', ''))

        numpy_feature = np.load('./CLIP_features/' + f)
        torch_feature = torch.from_numpy(numpy_feature)

        video_features = torch.vstack((video_features, torch_feature))
        
        for fid in frame_id:
            frame_ids.append(fid)
            video_ids.append(video_id)

def get_random_video_ids(n:int = 30):
    global video_ids
    sample_ids = np.random.randint(0, len(video_ids), size=n)
    sample_names = [video_ids[i] for i in sample_ids]
    return sample_names

@app.route('/')
def home_page():
    return render_template('text_query.html', names=get_random_video_ids(), captions=captions)

@app.route('/export')
def export_result():
    global video_ids
    global result_indices
    global frame_ids
    result_video_ids = np.array(video_ids)[result_indices].tolist()
    result_frame_ids = np.array(frame_ids)[result_indices].tolist()

    for i in range(len(result_video_ids)):
        result_video_ids[i] = result_video_ids[i] + '.mp4'

    df = pd.DataFrame({
        'video_id': result_video_ids,
        'frame_id': result_frame_ids
    })

    df.to_csv('./results/sample.csv', index=False, header=False)

    return redirect(url_for('home_page'))

@app.route('/text', methods=['GET', 'POST'])
def text_query():
    try:
        vids = request.args.getlist('vids')
    except:
        vids = get_random_video_ids(n=30)
        print('No return videos')

    global captions

    return render_template('text_query.html', names=vids, captions=captions)

@app.route('/query', methods=['POST'])
def add_caption():
    captions.append(request.form['caption'])

    compute_results(caption=captions[-1], n_retrieved=100)

    global video_ids
    global result_indices
    global frame_ids
    result_video_ids = np.array(video_ids)[result_indices].tolist()
    result_frame_ids = np.array(frame_ids)[result_indices].tolist()

    return redirect(url_for('text_query', vids=result_video_ids[:30]))

@app.route('/videokis')
def video_kis():
    vids = request.args.getlist('vids')
    if len(vids) == 0:
        vids = get_random_video_ids(n=30)
        print('No query video')

    return render_template('video_query.html', names=vids)

@app.route('/videokis', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        global videokis_filename_current
        videokis_filename_current = filename

        # print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        return render_template('video_query.html', filename=filename, names=get_random_video_ids(n=30))

@app.route('/videokis/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/videokis/query', methods=["POST"])
def video_kis_query():
    print('Enter function video_kis_query')
    global videokis_filename_current
    print(videokis_filename_current)
    cap = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], videokis_filename_current))

    if cap.isOpened()== False:
        print("Error opening video file")

    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            im_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(im_cv2)

            img = clip_preprocessor(im_pil)
            images.append(img)
        else:
            cap.release()
            cv2.destroyAllWindows()
            break

    image_input = torch.tensor(np.stack(images)).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        image_features = F.normalize(image_features)

    result_feature = torch.mean(image_features, dim=0)

    compute_results_video(result_feature, n_retrieved=100)

    global video_ids
    global result_indices
    global frame_ids
    result_video_ids = np.array(video_ids)[result_indices].tolist()
    result_frame_ids = np.array(frame_ids)[result_indices].tolist()

    return redirect(url_for('video_kis', vids=result_video_ids[:30]))

    
def compute_results_video(feature, n_retrieved=100):
    global result_indices
    global video_ids
    global frame_ids
    
    if len(result_indices) > 0:
        image_features = video_features[result_indices]
    else:
        image_features = video_features

    with torch.no_grad():
        # normalize
        image_features = F.normalize(image_features)

        # compute cosine similarity score between text feature and all image features in result_ids
        cos_similarity = image_features @ feature.T
        sorted_indices = torch.topk(cos_similarity, n_retrieved, dim=0, largest=True).indices.cpu()
        result_indices = sorted_indices.numpy().flatten().tolist()

@app.route('/videos/<string:video_id>', methods=['POST', 'GET'])
def get_video(video_id:str):
    return send_from_directory(server_base_path / 'videos', video_id + '.mp4')

def compute_results(caption, n_retrieved=100):
    global result_indices
    global video_ids
    global frame_ids
    
    if len(result_indices) > 0:
        image_features = video_features[result_indices]
    else:
        image_features = video_features

    with torch.no_grad():
        text_input = clip.tokenize(caption, truncate=True).to(device)
        text_feature = clip_model.encode_text(text_input)

        # normalize
        image_features = F.normalize(image_features)
        text_feature   = F.normalize(text_feature)

        # compute cosine similarity score between text feature and all image features in result_ids
        cos_similarity = image_features @ text_feature.T
        sorted_indices = torch.topk(cos_similarity, n_retrieved, dim=0, largest=True).indices.cpu()
        result_indices = sorted_indices.numpy().flatten().tolist()

@app.route('/clear')
def clear_all():
    result_indices.clear()
    captions.clear()

    print(result_indices)

    return redirect('/')

def delete_uploaded_images():
    '''
    For privacy reasons delete the uploaded images after 500 seconds
    '''
    FILE_LIFETIME = 500
    SLEEP_TIME = 50
    while True:
        for iter_path in app.config['UPLOAD_FOLDER'].rglob('*'):
            if iter_path.is_file():
                if time.time() - iter_path.stat().st_mtime > FILE_LIFETIME:
                    iter_path.unlink()

        time.sleep(SLEEP_TIME)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)