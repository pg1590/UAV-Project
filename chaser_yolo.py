import os
os.environ["ULTRALYTICS_OFFLINE"] = "True"
os.environ["QT_QPA_PLATFORM"] = "xcb"

from djitellopy import Tello
import cv2
import numpy as np
import time
import csv
from ultralytics import YOLO

# ---------------- YOLO MODEL ----------------
MODEL_PATH = "model2/yolov8s_rc_car/weights/best.pt"
model = YOLO(MODEL_PATH)

# ---------------- PARAMETERS ----------------
REAL_WIDTH = 0.22   # same assumption as before (object width)
FOCAL_LENGTH = 466
DESIRED_DISTANCE = 1.0

FRAME_W = 480
FRAME_H = 360

CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

TARGET_PIXEL_Y = CENTER_Y + 30

# ---------------- PID ----------------
class PID:
    def __init__(self,kp,ki,kd,limit=40):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.limit = limit

    def update(self,error,dt):
        self.integral += error*dt
        derivative = (error-self.prev_error)/dt if dt>0 else 0
        self.prev_error = error

        output = self.kp*error + self.ki*self.integral + self.kd*derivative
        return int(np.clip(output,-self.limit,self.limit))


yaw_pid = PID(0.25,0.0,0.08,40)
vert_pid = PID(0.35,0.0,0.12,30)
fwd_pid  = PID(25,0.0,3,40)

# ---------------- KALMAN FILTER ----------------
class KalmanFilter:
    def __init__(self):
        self.state = np.zeros((4,1))
        self.P = np.eye(4)*500

        self.F = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]
        ])

        self.H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        self.R = np.eye(2)*5
        self.Q = np.eye(4)*0.01

    def update(self,measurement):
        z = np.array(measurement).reshape(2,1)

        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4)-K@self.H)@self.P

        return int(self.state[0]),int(self.state[1])


kf = KalmanFilter()

# ---------------- UTILITIES ----------------
def estimate_depth(focal_length,real_width,pixel_width):
    if pixel_width <= 0:
        return None
    return (focal_length*real_width)/pixel_width

def create_tracker():
    if int(cv2.__version__.split('.')[0])>=4:
        return cv2.TrackerCSRT_create()
    else:
        return cv2.legacy.TrackerCSRT_create()

# ---------------- MAIN ----------------
def main():

    log_data = []
    start_time = time.time()

    tello = Tello()

    try:
        tello.connect()
        print("Battery:",tello.get_battery())

        tello.streamoff()
        time.sleep(1)
        tello.streamon()

        frame_reader = tello.get_frame_read()

        while frame_reader.frame is None:
            time.sleep(0.05)

        tello.takeoff()
        time.sleep(1)
        tello.move_up(20)
        time.sleep(2)

        tracker = None
        tracking = False

        last_box = None
        last_depth = None

        frame_count = 0
        prev_time = time.time()

        while True:

            frame = frame_reader.frame
            if frame is None:
                continue

            frame = cv2.resize(frame,(FRAME_W,FRAME_H))
            display = frame.copy()

            frame_count += 1

            now = time.time()
            dt = now-prev_time
            prev_time = now

            yaw_cmd = 0
            up_cmd = 0
            forward_cmd = 0

            # ---------------- YOLO DETECTION ----------------
            if frame_count % 4 == 0:

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb, imgsz=320, verbose=False)

                if results and len(results[0].boxes) > 0:

                    best_box = max(results[0].boxes,
                                   key=lambda b: float(b.conf))

                    conf = float(best_box.conf)

                    if conf > 0.5:

                        x1,y1,x2,y2 = map(int,best_box.xyxy[0])

                        w = x2-x1
                        h = y2-y1

                        tracker = create_tracker()
                        tracker.init(frame,(x1,y1,w,h))

                        tracking = True
                        last_box = (x1,y1,w,h)

                        last_depth = estimate_depth(
                            FOCAL_LENGTH,
                            REAL_WIDTH,
                            h   # use height
                        )

            # ---------------- TRACKER ----------------
            if tracking and tracker is not None:

                success,bbox = tracker.update(frame)

                if success:

                    x,y,w,h = map(int,bbox)

                    u = x + w//2
                    v = y + h//2

                    u,v = kf.update([u,v])

                    last_box = (u-w//2,v-h//2,w,h)

                    last_depth = estimate_depth(
                        FOCAL_LENGTH,
                        REAL_WIDTH,
                        h
                    )

                else:
                    tracking = False

            # ---------------- CONTROL ----------------
            if last_box is not None and last_depth is not None:

                x,y,w,h = last_box

                u = x + w//2
                v = y + h//2

                error_x = u - CENTER_X
                error_y = v - TARGET_PIXEL_Y
                error_z = last_depth - DESIRED_DISTANCE

                yaw_cmd = yaw_pid.update(error_x,dt)
                up_cmd = vert_pid.update(-error_y,dt)

                DEPTH_TOL = 0.35
                if abs(error_z) < DEPTH_TOL:
                    forward_cmd = 0
                else:
                    forward_cmd = fwd_pid.update(error_z, dt)

                forward_cmd = np.clip(forward_cmd, -40, 40)

                # -------- LOGGING --------
                current_time = time.time() - start_time
                log_data.append([
                    current_time,
                    error_x,
                    error_y,
                    error_z,
                    yaw_cmd,
                    up_cmd,
                    forward_cmd,
                    last_depth,
                    1
                ])

                # -------- VISUALS --------
                cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.circle(display,(u,v),4,(0,255,0),-1)

                cv2.putText(display,f"Depth:{last_depth:.2f}m",
                            (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                cv2.putText(display,
                            f"ex:{error_x:.1f} ey:{error_y:.1f} ez:{error_z:.2f}",
                            (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0,255,255),
                            2)

            else:
                tello.send_rc_control(0,0,0,0)

            # FPS
            fps = 1/dt if dt>0 else 0
            cv2.putText(display,f"FPS:{fps:.1f}",(10,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

            cv2.circle(display,(CENTER_X,CENTER_Y),4,(255,0,0),-1)

            cv2.imshow("Chaser Drone",display)

            tello.send_rc_control(0,int(forward_cmd),int(up_cmd),int(yaw_cmd))

            key = cv2.waitKey(1)&0xFF

            if key == ord('q'):
                print("Emergency landing triggered")
                try:
                    tello.send_rc_control(0,0,0,0)
                    time.sleep(0.2)
                    tello.land()
                    time.sleep(2)
                except Exception as e:
                    print("Landing error:", e)
                break

            time.sleep(0.03)

        tello.send_rc_control(0,0,0,0)

    finally:

        tello.streamoff()
        cv2.destroyAllWindows()

        with open("flight_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time","error_x","error_y","error_z",
                "yaw_cmd","up_cmd","forward_cmd","depth","tracking"
            ])
            writer.writerows(log_data)

        print("Log saved!")
        print("Battery:",tello.get_battery())
        tello.end()

if __name__ == "__main__":
    main()