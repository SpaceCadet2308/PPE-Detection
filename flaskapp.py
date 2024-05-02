from flask import Flask, render_template, Response, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
from YOLO_Video import video_detection

app = Flask(__name__)
app.config["SECRET_KEY"] = "skhamzahharmo"
app.config["UPLOAD_FOLDER"] = "static/files"

def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
def generate_frames_web():
    # OpenCV code to capture frames from the webcam
    video_capture = cv2.VideoCapture(0)  # Assuming webcam is at index 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


# def generate_frames_web():
#     # OpenCV code to capture frames from the webcam
#     video_capture = cv2.VideoCapture(0)  # Assuming webcam is at index 0
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         ret, buffer = cv2.imencode(".jpg", frame)
#         frame = buffer.tobytes()
#         yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    session.clear()
    return render_template("PPE/home_ppe.html")


@app.route("/liveweb_ppe.html", methods=["GET", "POST"])
def webcam():
    session.clear()
    return render_template("PPE/liveweb_ppe.html")


@app.route("/vp_ppe.html", methods=["GET", "POST"])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(
            os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        )
        session["video_path"] = os.path.join(
            app.config["UPLOAD_FOLDER"], secure_filename(file.filename)
        )
    return render_template("PPE/vp_ppe.html", form=form)


@app.route("/video")
def video():
    return Response(
        generate_frames(path_x=session.get("video_path", None)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/webapp")
def webapp():
    return Response(
        generate_frames_web(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
