import os
import time
import glob
import io
import multiprocessing
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from PIL import Image
from pathlib import Path
from typing import List, Generator, Union, Dict, Any
import numpy as np
import logging


# The render server exists to serve images written to /dev/shm
# by workers over a web interface for debugging purposes


RENDER_ROOT = "/dev/shm/render/"
CLIENT_LOCK = Path(f"{RENDER_ROOT}/client_conn.lock")
conn_clients = 0
ACTION_QUEUE: Union[multiprocessing.Queue, None]
RESPONSE_QUEUE: Union[multiprocessing.Queue, None]

app = Flask(__name__)
socketio = SocketIO(app)


@socketio.on("action_input", namespace="/")
def action_input(msg) -> None:
    global ACTION_QUEUE, RESPONSE_QUEUE

    char = chr(msg["data"])
    if not ACTION_QUEUE:
        print("Action queue does not exist, interactive control does not work")
    ACTION_QUEUE.put(char)  # type: ignore

    env_data: Dict[str, Any] = RESPONSE_QUEUE.get()
    emit("env_response", env_data)


# SocketIO functions use a lock file for a web client connection
# This ensures we are not wasting cycles producing images if there
# are no viewers
@socketio.on("connect", namespace="/")
def connect() -> None:
    global conn_clients
    conn_clients += 1
    print(f"Client connected, clients: {conn_clients}")
    if conn_clients >= 0:
        print("Enabling visualization")
        CLIENT_LOCK.touch()


@socketio.on("disconnect", namespace="/")
def disconnect() -> None:
    global conn_clients
    conn_clients -= 1
    print(f"Client disconnected, clients: {conn_clients}")
    if conn_clients <= 0:
        print("Disabling visualization")
        CLIENT_LOCK.unlink(missing_ok=True)


# Default load page
@app.route("/")
def index():
    return render_template("index.html")


def concat_v_from_paths(img_paths: List[str]) -> np.ndarray:
    imgs = [Image.open(p) for p in img_paths]
    dst = Image.new("RGB", (imgs[0].width, imgs[0].height * len(imgs)))
    for i in range(len(imgs)):
        dst.paste(imgs[i], (0, imgs[0].height * i))
    return dst


def concat_v_from_imgs(imgs: List[np.ndarray]) -> np.ndarray:
    dst = Image.new("RGB", (imgs[0].width, imgs[0].height * len(imgs)))
    for i in range(len(imgs)):
        dst.paste(imgs[i], (0, imgs[0].height * i))
    return dst


def concat_h_from_paths(img_paths: List[str]) -> np.ndarray:
    imgs = [Image.open(p) for p in img_paths]
    dst = Image.new(
        "RGB", (sum(img.width for img in imgs), max(img.height for img in imgs))
    )
    for i in range(len(imgs)):
        if i == 0:
            hpos = 0
        else:
            hpos = sum(img.width for img in imgs[:i])
        dst.paste(imgs[i], (hpos, 0))
    return dst


def gen() -> Generator[bytes, None, None]:
    while True:
        try:
            if os.path.isdir(RENDER_ROOT):
                render_dirs = sorted(
                    [f.path for f in os.scandir(RENDER_ROOT) if f.is_dir()]
                )
                # TODO: Select any proc, not just the first
                render_dirs = [render_dirs[0]]
                img_paths = [sorted(glob.glob(f"{d}/*.jpg")) for d in render_dirs]
            if any(img_paths):
                # Concatenate images in the same dir (rgb, graph, etc) horizontally
                # and concatenate different dirs (worker1, worker2, etc) vertically
                h_imgs = [concat_h_from_paths(d) for d in img_paths]
                final_img = concat_v_from_imgs(h_imgs)
                buf = io.BytesIO()
                final_img.save(buf, format="jpeg")
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n"
                )
        except Exception as e:
            print("Render server failed to serve image")
            print(e)


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main(action_queue=None, response_queue=None):
    logging.getLogger("socketio").setLevel(logging.ERROR)
    logging.getLogger("engineio").setLevel(logging.ERROR)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    global ACTION_QUEUE, RESPONSE_QUEUE
    ACTION_QUEUE = action_queue
    RESPONSE_QUEUE = response_queue
    socketio.run(app, host="0.0.0.0", debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
