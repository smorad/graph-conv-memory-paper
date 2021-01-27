import os
import time
import glob
import io
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from PIL import Image
from pathlib import Path
from typing import List 
import numpy as np
import logging


# The render server exists to serve images written to /dev/shm
# by workers over a web interface for debugging purposes


RENDER_ROOT = '/dev/shm/render/'
CLIENT_LOCK = Path(f'{RENDER_ROOT}/client_conn.lock')
conn_clients = 0

app = Flask(__name__)
socketio = SocketIO(app)

# SocketIO functions use a lock file for a web client connection
# This ensures we are not wasting cycles producing images if there
# are no viewers
@socketio.on("connect", namespace="/")
def connect() -> None:
    global conn_clients
    conn_clients += 1
    print(f'Client connected, clients: {conn_clients}')
    if conn_clients >= 0:
        print('Enabling visualization')
        CLIENT_LOCK.touch()

@socketio.on("disconnect", namespace="/")
def disconnect() -> None:
    global conn_clients
    conn_clients -= 1
    print(f'Client disconnected, clients: {conn_clients}')
    if conn_clients <= 0:
        print('Disabling visualization')
        CLIENT_LOCK.unlink(missing_ok=True)

# Default load page
@app.route('/')
def index():
    return render_template('index.html')

def concat_v_from_paths(img_paths: List[str]) -> np.ndarray:
    imgs = [Image.open(p) for p in img_paths] 
    dst = Image.new('RGB', (imgs[0].width, imgs[0].height  * len(imgs)))
    for i in range(len(imgs)):
        dst.paste(imgs[i], (0, imgs[0].height * i))
    return dst

def concat_v_from_imgs(imgs: List[np.ndarray]) -> np.ndarray:
    dst = Image.new('RGB', (imgs[0].width, imgs[0].height  * len(imgs)))
    for i in range(len(imgs)):
        dst.paste(imgs[i], (0, imgs[0].height * i))
    return dst

def concat_h_from_paths(img_paths: List[str]) -> np.ndarray:
    imgs = [Image.open(p) for p in img_paths] 
    dst = Image.new('RGB', (sum(img.width for img in imgs), max(img.height for img in imgs)))
    for i in range(len(imgs)):
        if i == 0:
            hpos = 0
        else:
            hpos = sum(img.width for img in imgs[:i])
        dst.paste(imgs[i], (hpos, 0))
    return dst

def gen() -> str:
    while True:
        if not os.path.isdir(RENDER_ROOT):
            continue
        render_dirs = [f.path for f in os.scandir(RENDER_ROOT) if f.is_dir()]
        img_paths = [sorted(glob.glob(f'{d}/*.jpg')) for d in render_dirs]
        if not any(img_paths):
            continue
        # Concatenate images in the same dir (rgb, graph, etc) horizontally
        # and concatenate different dirs (worker1, worker2, etc) vertically
        h_imgs = [concat_h_from_paths(d) for d in img_paths]
        final_img = concat_v_from_imgs(h_imgs)
        buf = io.BytesIO()
        final_img.save(buf, format='jpeg')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.getvalue() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    logging.getLogger('socketio').setLevel(logging.ERROR)
    logging.getLogger('engineio').setLevel(logging.ERROR)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    socketio.run(app, host='0.0.0.0', debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
