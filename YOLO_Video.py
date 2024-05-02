from ultralytics import YOLO
import cv2
import math
import telepot

# Initialize Telegram bot
bot_token = "6995379170:AAHmWizulJoeQDx6HPfVphwmA9pMxgxCj5c"
chat_id = "1691301619"
bot = telepot.Bot(token=bot_token)


def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    model = YOLO("templates/PPE/ppe.pt")
    classNames = [
        "Hardhat",
        "Mask",
        "NO-Hardhat",
        "NO-Mask",
        "NO-Safety Vest",
        "Person",
        "Safety Cone",
        "Safety Vest",
        "machinery",
        "vehicle",
    ]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f"{class_name}{conf}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name in [
                    "Hardhat",
                    "Mask",
                    "Person",
                    "Safety Cone",
                    "Safety Vest",
                ]:
                    color = (0, 255, 0)  # Green
                elif class_name in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]:
                    color = (0, 0, 255)  # Red
                elif class_name in ["machinery", "vehicle"]:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 0, 0)  # Default to Blue
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 2),
                    0,
                    1,
                    [255, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                if (
                    class_name in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]
                    and conf > 0.8
                ):
                    # Send message to Telegram
                    message = (
                        f"{class_name} Detected in the frame with confidence {conf}"
                    )
                    bot.sendMessage(chat_id=chat_id, text=message)
        yield img


# Main loop
if __name__ == "__main__":
    # Replace 'video_file_path' with the path to your video file
    video_gen = video_detection("video_file_path")
    for frame in video_gen:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
