from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import sys

sys.path.append(os.path.abspath("src"))

from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def generate_outputs(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256,256))

    outputs = []

    def save(img, name):
        path = os.path.join(OUTPUT_FOLDER, f"{name}.png")
        cv2.imwrite(path, img)
        outputs.append(path)

    save(gray, "1_original")
    save(resized, "2_resized")
    save(cv2.Canny(resized,50,150), "3_edges")
    save(cv2.threshold(resized,127,255,cv2.THRESH_BINARY)[1], "4_threshold")

    heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
    save(heatmap, "5_heatmap")

    blend = cv2.addWeighted(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR),0.6,heatmap,0.4,0)
    save(blend, "6_blend")

    save(cv2.GaussianBlur(resized,(15,15),0), "7_blur")

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    save(cv2.filter2D(resized,-1,kernel), "8_sharpen")

    return outputs


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")


@app.route('/predict', methods=['POST'])
def predict():

    file1 = request.files['image1']
    file2 = request.files['image2']

    path1 = os.path.join(UPLOAD_FOLDER, "img1_" + file1.filename)
    path2 = os.path.join(UPLOAD_FOLDER, "img2_" + file2.filename)

    file1.save(path1)
    file2.save(path2)

    result1 = predict_image(path1)
    result2 = predict_image(path2)

    outputs1 = generate_outputs(path1)
    outputs2 = generate_outputs(path2)

    return render_template(
    "result.html",

    # MAIN DISPLAY (use image1 as main)
    image_path=path1,
    outputs=outputs1,
    result=result1,
    label=result1["label"],
    confidence=result1["confidence"],

    # COMPARISON DATA
    img1=path1,
    img2=path2,
    result1=result1,
    result2=result2
)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)