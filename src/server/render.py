import os
import time
import glob
import io
from flask import Flask, render_template, Response
from PIL import Image

app = Flask(__name__)

RENDER_ROOT = '/dev/shm/render/'

@app.route('/')
def index():
    return render_template('index.html')


def concat_v_from_paths(img_paths):
    imgs = [Image.open(p) for p in img_paths] 
    dst = Image.new('RGB', (imgs[0].width, imgs[0].height  * len(imgs)))
    for i in range(len(imgs)):
        dst.paste(imgs[i], (0, imgs[0].height * i))
    return dst

def concat_v_from_imgs(imgs):
    dst = Image.new('RGB', (imgs[0].width, imgs[0].height  * len(imgs)))
    for i in range(len(imgs)):
        dst.paste(imgs[i], (0, imgs[0].height * i))
    return dst

def concat_h_from_paths(img_paths):
    imgs = [Image.open(p) for p in img_paths] 
    dst = Image.new('RGB', (sum(img.width for img in imgs), max(img.height for img in imgs)))
    for i in range(len(imgs)):
        if i == 0:
            hpos = 0
        else:
            hpos = sum(img.width for img in imgs[:i])
        dst.paste(imgs[i], (hpos, 0))
    return dst

def gen():
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
