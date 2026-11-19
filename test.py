import math
import time
from pathlib import Path
from collections import deque
import cv2
import mediapipe as mp

import serial
from serial.tools import list_ports
import numpy as np

arduino = None
arduino_status = "Arduino: not detected"


def find_arduino_port(ports):
    mac_ports = [port for port in ports if port.device.startswith("/dev/cu.")]
    candidates = mac_ports or list(ports)
    arduino_ports = []

    for port in candidates:
        description = " ".join(
            value or ""
            for value in (port.description, port.manufacturer, port.product)
        ).lower()
        if port.vid in (0x2341, 0x2A03) or "arduino" in description:
            arduino_ports.append(port)

    if len(arduino_ports) == 1:
        return arduino_ports[0]
    if arduino_ports:
        return None

    usb_serial_markers = (
        "usbmodem",
        "usbserial",
        "wchusbserial",
        "slab_usbtouart",
        "usbtouart",
    )
    usb_serial_ports = [
        port for port in candidates
        if any(marker in port.device.lower() for marker in usb_serial_markers)
    ]
    return usb_serial_ports[0] if len(usb_serial_ports) == 1 else None


serial_ports = list(list_ports.comports())
arduino_port = find_arduino_port(serial_ports)
if arduino_port is None:
    available_ports = ", ".join(port.device for port in serial_ports) or "none"
    print(f"Arduino not detected. Available serial ports: {available_ports}")
else:
    try:
        arduino = serial.Serial(
            arduino_port.device,
            9600,
            timeout=0,
            write_timeout=0.12,
        )

        # Opening the port usually resets the Uno.
        time.sleep(2)
        arduino.reset_input_buffer()
    except serial.SerialException as exc:
        if arduino is not None and arduino.is_open:
            arduino.close()
        arduino = None
        arduino_status = "Arduino: found but connection failed"
        print(f"Could not connect to Arduino on {arduino_port.device}: {exc}")
    else:
        arduino_status = f"Arduino: connected ({arduino_port.device})"
        print(arduino_status)

busy = False
serial_buffer = ""

# --- Gantry settings ---
BED_X_MM = 200.0            # usable travel in X
BED_Y_MM = 200.0            # usable travel in Y
START_X_MM = BED_X_MM / 2   # where the gantry physically is when the
START_Y_MM = BED_Y_MM / 2   # script starts (default: the middle)

# --- Hand-size ruler settings ---
REACH_X = 4.0               # hand units from the bed centre to its left/right edge
REACH_Y = 3.0               # hand units from the bed centre to its top/bottom edge
RULER_SMOOTHING = 0.05      # how quickly the ruler adapts to distance changes
TARGET_SMOOTHING = 0.3      # 0-1: lower = smoother but slower to react
HAND_LOST_SECONDS = 0.5     # tracking dropouts shorter than this are ignored

# --- Motion settings ---
MAX_STEP_MM = 5.0           # longest single move sent to the Arduino
MIN_STEP_MM = 0.5           # ignore differences smaller than this (stops jitter)

MM_PER_HAND_X = BED_X_MM / (2 * REACH_X)
MM_PER_HAND_Y = BED_Y_MM / (2 * REACH_Y)

# Gantry state, tracked in Python because the Arduino takes relative moves.
gantry_x, gantry_y = START_X_MM, START_Y_MM   # position after finished moves
pending_dx = pending_dy = 0.0                 # the move in progress
target_x, target_y = START_X_MM, START_Y_MM   # where the hand wants it

# Ruler state.
anchor = None       # [x, y] pixel position that maps to `origin`
origin = None       # (x, y) gantry position in mm that the anchor maps to
ruler = None        # smoothed hand size in pixels
last_hand_time = 0.0
# --- Snake (gantry trail) settings ---
SNAKE_LENGTH = 60         # how many past positions the snake keeps
SNAKE_MIN_MOVE_MM = 1.0   # add a new segment only after moving this far
MAP_WIDTH = 200           # mini-map width in pixels
MAP_MARGIN = 20           # gap between the mini-map and the frame edge

snake = deque(maxlen=SNAKE_LENGTH)
last_servo_command = None
candidate_command = None
candidate_since = time.monotonic()
GESTURE_HOLD_SECONDS = 0.15

MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")

options = mp.tasks.vision.GestureRecognizerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
)


# Open palm detection code block
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def joint_angle(a, b, c):
    """Angle at landmark b, in degrees."""
    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def is_open_hand(landmarks):
    # Use image pixel proportions so x/y have the same scale.
    points = np.array([
        [point.x * width, point.y * height]
        for point in landmarks
    ])

    # MCP, PIP, DIP, TIP for the four fingers.
    fingers = [
        (5, 6, 7, 8),      # Index
        (9, 10, 11, 12),   # Middle
        (13, 14, 15, 16),  # Ring
        (17, 18, 19, 20),  # Pinky
    ]

    for mcp, pip, dip, tip in fingers:
        pip_angle = joint_angle(
            points[mcp], points[pip], points[dip]
        )
        dip_angle = joint_angle(
            points[pip], points[dip], points[tip]
        )

        if pip_angle < 155 or dip_angle < 150:
            return False

    return True
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


camera = cv2.VideoCapture(0)

try:
    with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
        previous_timestamp = -1

        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                break

            # Mirror the image for intuitive interaction.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode requires increasing timestamps in milliseconds.
            timestamp = max(
                previous_timestamp + 1,
                time.monotonic_ns() // 1_000_000
            )
            previous_timestamp = timestamp


            # Get results from camera
            # !!!
            result = recognizer.recognize_for_video(image, timestamp)
            # !!!

            height, width = frame.shape[:2]

            # Servo motor code block
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            height, width = frame.shape[:2]
            servo_command = None

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                if is_open_hand(hand):
                    servo_command = "H"  # Open servo
                elif result.gestures and result.gestures[0]:
                    gesture = result.gestures[0][0]

                    if (
                        gesture.score >= 0.6
                        and gesture.category_name in (
                            "Closed_Fist",
                            "Thumb_Up",
                            "Thumb_Down",
                        )
                    ):
                        servo_command = "L"  # Close servo


            # if result.gestures and result.gestures[0]:
            #     gesture = result.gestures[0][0]

            #     if gesture.score >= 0.6:
            #         if gesture.category_name == "Open_Palm":
            #             servo_command = "H"
            #         elif (
            #             gesture.category_name == "Closed_Fist"
            #             or gesture.category_name == "Thumbs_Up"
            #             or gesture.category_name == "Thumbs_Down"
            #         ):
            #             servo_command = "L"

            # Require a stable gesture before changing the servo target.
            now = time.monotonic()
            if servo_command != candidate_command:
                candidate_command = servo_command
                candidate_since = now

            if (
                candidate_command is not None
                and now - candidate_since >= GESTURE_HOLD_SECONDS
                and candidate_command != last_servo_command
                and arduino is not None
            ):
                arduino.write((candidate_command + "\n").encode())
                last_servo_command = candidate_command
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



            # Hand-size ruler code block: hand position -> gantry target
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                wrist, palm = hand[0], hand[9]

                # Work in pixels so x and y have the same scale.
                px, py = palm.x * width, palm.y * height
                size = math.hypot(
                    px - wrist.x * width,
                    py - wrist.y * height,
                )
                last_hand_time = now

                if size > 1:
                    if anchor is None:
                        # The hand's current position maps to wherever
                        # the gantry already is, so nothing jumps.
                        anchor = [px, py]
                        origin = (target_x, target_y)
                        ruler = size
                    else:
                        ruler += RULER_SMOOTHING * (size - ruler)

                    # Offset from the anchor in hand units, converted to mm.
                    raw_x = origin[0] + (px - anchor[0]) / ruler * MM_PER_HAND_X
                    raw_y = origin[1] + (anchor[1] - py) / ruler * MM_PER_HAND_Y

                    # Edge push: past the bed edge, drag the anchor along.
                    if raw_x > BED_X_MM:
                        anchor[0] += (raw_x - BED_X_MM) / MM_PER_HAND_X * ruler
                        raw_x = BED_X_MM
                    elif raw_x < 0:
                        anchor[0] += raw_x / MM_PER_HAND_X * ruler
                        raw_x = 0.0

                    if raw_y > BED_Y_MM:
                        anchor[1] -= (raw_y - BED_Y_MM) / MM_PER_HAND_Y * ruler
                        raw_y = BED_Y_MM
                    elif raw_y < 0:
                        anchor[1] -= raw_y / MM_PER_HAND_Y * ruler
                        raw_y = 0.0

                    target_x += TARGET_SMOOTHING * (raw_x - target_x)
                    target_y += TARGET_SMOOTHING * (raw_y - target_y)

            elif anchor is not None and now - last_hand_time > HAND_LOST_SECONDS:
                # Hand is gone: forget the anchor and stop where we are.
                anchor = None
                origin = None
                ruler = None
                target_x = gantry_x + pending_dx
                target_y = gantry_y + pending_dy
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



            # Nema 23 motors code block
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if arduino is not None:
                # Read Arduino responses without blocking the camera.
                if arduino.in_waiting:
                    serial_buffer += arduino.read(
                        arduino.in_waiting
                    ).decode(errors="ignore")

                    while "\n" in serial_buffer:
                        response, serial_buffer = serial_buffer.split("\n", 1)
                        response = response.strip()

                        if response == "Done.":
                            gantry_x += pending_dx
                            gantry_y += pending_dy
                            pending_dx = pending_dy = 0.0
                            busy = False
                        elif response == "Stopped; movement incomplete.":
                            # Position is now uncertain; this script only
                            # sends a stop when quitting.
                            pending_dx = pending_dy = 0.0
                            busy = False

                # Send one small relative move toward the target,
                # then wait for the Arduino to finish it.
                if not busy:
                    dx = target_x - gantry_x
                    dy = target_y - gantry_y
                    distance = math.hypot(dx, dy)

                    if distance >= MIN_STEP_MM:
                        if distance > MAX_STEP_MM:
                            dx *= MAX_STEP_MM / distance
                            dy *= MAX_STEP_MM / distance

                        dx, dy = round(dx, 2), round(dy, 2)
                        arduino.write(f"{dx:.2f} {dy:.2f}\n".encode())
                        pending_dx, pending_dy = dx, dy
                        busy = True
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


            label = "No hand"

            if result.hand_landmarks:
                label = "Unknown gesture"

                # Draw the detected hand landmarks.
                height, width = frame.shape[:2]
                for landmark in result.hand_landmarks[0]:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                if result.gestures and result.gestures[0]:
                    gesture = result.gestures[0][0]

                    # Display gesture name if confidence > 60%
                    if gesture.score >= 0.6:
                        label = (
                            f"{gesture.category_name} "
                            f"({gesture.score:.0%})"
                        )

                hand = result.hand_landmarks[0]

            # Draw the area of the image that currently maps to the bed.
            if anchor is not None:
                left = anchor[0] - origin[0] / MM_PER_HAND_X * ruler
                right = anchor[0] + (BED_X_MM - origin[0]) / MM_PER_HAND_X * ruler
                top = anchor[1] - (BED_Y_MM - origin[1]) / MM_PER_HAND_Y * ruler
                bottom = anchor[1] + origin[1] / MM_PER_HAND_Y * ruler
                cv2.rectangle(
                    frame,
                    (int(left), int(top)),
                    (int(right), int(bottom)),
                    (255, 200, 0), 2
                )

            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 125, 255), 2
            )

            cv2.putText(
                frame,
                f"Target: {target_x:.0f}, {target_y:.0f} mm",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 200, 0), 2
            )

            status_color = (0, 255, 0) if arduino is not None else (0, 125, 255)
            cv2.putText(
                frame,
                arduino_status,
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

            # Snake code block: trail of the gantry on a mini-map of the bed
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if arduino is not None:
                snake_x, snake_y = gantry_x, gantry_y
            else:
                # No Arduino: follow the target so it can be previewed.
                snake_x, snake_y = target_x, target_y

            if (
                not snake
                or math.hypot(snake_x - snake[-1][0], snake_y - snake[-1][1])
                >= SNAKE_MIN_MOVE_MM
            ):
                snake.append((snake_x, snake_y))

            map_height = int(MAP_WIDTH * BED_Y_MM / BED_X_MM)
            map_left = width - MAP_WIDTH - MAP_MARGIN
            map_top = MAP_MARGIN

            # Darken the mini-map area and outline the bed.
            frame[map_top:map_top + map_height,
                  map_left:map_left + MAP_WIDTH] //= 3
            cv2.rectangle(
                frame,
                (map_left, map_top),
                (map_left + MAP_WIDTH, map_top + map_height),
                (255, 255, 255), 1
            )

            # Convert bed millimetres to mini-map pixels (bed Y points up).
            points = [
                (
                    int(map_left + x / BED_X_MM * MAP_WIDTH),
                    int(map_top + (1 - y / BED_Y_MM) * map_height),
                )
                for x, y in list(snake) + [(snake_x, snake_y)]
            ]

            # Body: thin at the tail, thick at the head.
            for i in range(1, len(points)):
                thickness = 1 + int(6 * i / len(points))
                cv2.line(frame, points[i - 1], points[i], (0, 200, 0), thickness)

            # Head: the gantry's current position.
            cv2.circle(frame, points[-1], 6, (0, 255, 0), -1)

            # Cross: where the hand is asking the gantry to go.
            cv2.drawMarker(
                frame,
                (
                    int(map_left + target_x / BED_X_MM * MAP_WIDTH),
                    int(map_top + (1 - target_y / BED_Y_MM) * map_height),
                ),
                (0, 0, 255), cv2.MARKER_CROSS, 10, 2
            )
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            cv2.imshow("Press Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    try:
        if arduino is not None and arduino.is_open:
            arduino.write(b"S\n")
            arduino.flush()
    finally:
        if arduino is not None:
            arduino.close()
        camera.release()
        cv2.destroyAllWindows()