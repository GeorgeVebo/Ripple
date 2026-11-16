import time
from pathlib import Path

import cv2
import mediapipe as mp

import serial

arduino = serial.Serial(
    "/dev/cu.usbmodem101",  # Replace with the port shown in Arduino IDE → Tools → Port.
    9600,
    timeout=0,
    write_timeout=0.12
)

# Opening the port usually resets the Uno.
time.sleep(2)
arduino.reset_input_buffer()

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


            # Servo motor code block
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            servo_command = None
            if result.gestures and result.gestures[0]:
                gesture = result.gestures[0][0]

                if gesture.score >= 0.6:
                    if gesture.category_name == "Open_Palm":
                        servo_command = "H"
                    elif gesture.category_name == "Closed_Fist":
                        servo_command = "L"

            # Require a stable gesture before changing the servo target.
            now = time.monotonic()

            if servo_command != candidate_command:
                candidate_command = servo_command
                candidate_since = now

            if (
                candidate_command is not None
                and now - candidate_since >= GESTURE_HOLD_SECONDS
                and candidate_command != last_servo_command
            ):
                arduino.write((candidate_command + "\n").encode())
                last_servo_command = candidate_command
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



            # Nema 23 motors code block
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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
                index_tip = hand[9]  # Landmark 9 = middle finger bottom

                x = int(index_tip.x * width / 10)
                y = int(index_tip.y * height / 10)

                cv2.putText(
                    frame,
                    f"Hand Position: ({x}, {y})",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 125, 255), 2
                )
            

            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 125, 255), 2
            )

            cv2.imshow("Press Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    try:
        if arduino.is_open:
            arduino.write(b"S\n")
            arduino.flush()
    finally:
        arduino.close()
        camera.release()
        cv2.destroyAllWindows()
