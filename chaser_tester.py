import os
os.environ["ULTRALYTICS_OFFLINE"] = "True"
os.environ["QT_QPA_PLATFORM"] = "xcb"

from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time

# ---------------- YOLO MODEL ----------------
MODEL_PATH = "train/weights/best.pt"
model = YOLO(MODEL_PATH)

# ---------------- PARAMETERS ----------------
FRAME_W = 640
FRAME_H = 480

CONF_THRESHOLD = 0.5

# ---------------- MAIN ----------------

def main():

    tello = Tello()

    try:

        # Connect
        tello.connect()
        print("Battery:", tello.get_battery())

        # Start stream
        tello.streamoff()
        time.sleep(1)

        tello.streamon()

        frame_reader = tello.get_frame_read()

        print("Waiting for stream...")

        while frame_reader.frame is None:
            time.sleep(0.05)

        print("Stream started!")

        while True:

            frame = frame_reader.frame

            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            display = frame.copy()

            # ---------------- YOLO DETECTION ----------------

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model(rgb, imgsz=320, verbose=False)

            if results and len(results[0].boxes) > 0:

                for box in results[0].boxes:

                    conf = float(box.conf)

                    if conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cls = int(box.cls[0])

                    label = model.names[cls]

                    # Draw bounding box
                    cv2.rectangle(display,
                                  (x1, y1),
                                  (x2, y2),
                                  (0,255,0),
                                  2)

                    # Draw label
                    cv2.putText(display,
                                f"{label} {conf:.2f}",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0,255,0),
                                2)

            # ---------------- DISPLAY ----------------

            cv2.imshow("YOLO Detection Test", display)

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord('q'):
                break

        print("Closing...")

    finally:

        tello.streamoff()
        cv2.destroyAllWindows()
        tello.end()


if __name__ == "__main__":
    main()