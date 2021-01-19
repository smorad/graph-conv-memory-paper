import os
import io
from flask import Flask, render_template, Response
from PIL import Image

app = Flask(__name__)

RENDER_ROOT = '/dev/shm/render/'

@app.route('/')
def index():
    return render_template('index.html')


def get_concat_v(img_paths):
    imgs = [Image.open(p) for p in img_paths] 
    dst = Image.new('RGB', (imgs[0].width, imgs[0].height  * len(imgs)))
    for i in range(len(imgs)):
        dst.paste(imgs[i], (0, imgs[0].height * i))
    return dst

def gen():
    while True:
        render_dirs = [f.path for f in os.scandir(RENDER_ROOT) if f.is_dir()]
        img_paths = [f'{d}/out.jpg' for d in render_dirs if os.path.isfile(f'{d}/out.jpg')]
        cat_img = get_concat_v(img_paths)
        buf = io.BytesIO()
        cat_img.save(buf, format='jpeg')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.getvalue() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
