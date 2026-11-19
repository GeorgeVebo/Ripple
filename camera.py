import time
from pathlib import Path

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
stop_sent = False
serial_buffer = ""

MOVE_MM = 1.0       # Small movement per command
DEAD_ZONE = 0.15    # Center neutral region

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

                        if response in ("Done.", "Stopped; movement incomplete."):
                            busy = False

                dx = 0.0
                dy = 0.0
                if result.hand_landmarks:
                    hand_point = result.hand_landmarks[0][9]

                    # Normalized coordinates: center is approximately (0.5, 0.5).
                    offset_x = hand_point.x - 0.5
                    offset_y = 0.5 - hand_point.y  # Up becomes positive Y

                    if abs(offset_x) > DEAD_ZONE:
                        dx = MOVE_MM if offset_x > 0 else -MOVE_MM

                    if abs(offset_y) > DEAD_ZONE:
                        dy = MOVE_MM if offset_y > 0 else -MOVE_MM

                if dx == 0 and dy == 0:
                    # Stop an ongoing move when centered or no hand is detected.
                    if busy and not stop_sent:
                        arduino.write(b"S\n")
                        stop_sent = True
                else:
                    # Send one small relative move, then wait for completion.
                    if not busy:
                        command = f"{dx:.2f} {dy:.2f}\n"
                        arduino.write(command.encode())
                        busy = True
                        stop_sent = False
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

            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 125, 255), 2
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
