#include <Servo.h>
#include <stdlib.h>
#include <math.h>

// ---------- Servo settings ----------
Servo motor;

const byte SERVO_PIN = 9;
const int CLOSE_PULSE = 180;
const int OPEN_PULSE = 20;

// ---------- CoreXY motor pins ----------
const byte A_PUL = 2;
const byte B_PUL = 3;
const byte A_DIR = 5;
const byte B_DIR = 6;

const bool INVERT_A = false;
const bool INVERT_B = false;

// Keep your calibrated value here.
// Example: 200 steps/rev, 1/16 stepping,
// 2 mm belt pitch, 20-tooth pulleys -> 80 steps/mm.
const float STEPS_PER_MM = 80.0;

const unsigned long STEP_INTERVAL_US = 2500;
const unsigned int PULSE_WIDTH_US = 20;

// ---------- Serial input ----------
char input[48];
byte inputLength = 0;
bool inputOverflow = false;

void setup() {
  // Common-anode TB6600 wiring: HIGH = idle.
  digitalWrite(A_PUL, HIGH);
  digitalWrite(B_PUL, HIGH);
  digitalWrite(A_DIR, HIGH);
  digitalWrite(B_DIR, HIGH);

  pinMode(A_PUL, OUTPUT);
  pinMode(B_PUL, OUTPUT);
  pinMode(A_DIR, OUTPUT);
  pinMode(B_DIR, OUTPUT);

  // Your existing servo initialization.
  motor.attach(SERVO_PIN);
  motor.write(CLOSE_PULSE);

  Serial.begin(9600);
  Serial.println("Enter relative X Y movement in mm.");
  Serial.println("H: servo open | L: servo close | S: stop gantry");
}

// Returns true if the character was a servo command.
bool handleServoCommand(char command) {
  if (command == 'H') {
    motor.write(OPEN_PULSE);
    return true;
  }
  else if (command == 'L') {
    motor.write(CLOSE_PULSE);
    return true;
  }

  return false;
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    // Handle single-character commands immediately.
    if (inputLength == 0 && !inputOverflow) {
      if (handleServoCommand(c)) continue;
      if (c == 'S' || c == 's') continue; // Already stationary
    }

    if (c == '\n' || c == '\r') {
      if (inputOverflow) {
        Serial.println("Input too long.");
      }
      else if (inputLength > 0) {
        input[inputLength] = '\0';
        processCommand(input);
      }

      inputLength = 0;
      inputOverflow = false;
    }
    else if (!inputOverflow) {
      if (inputLength < sizeof(input) - 1) {
        input[inputLength++] = c;
      }
      else {
        inputOverflow = true;
      }
    }
  }
}

void processCommand(char *text) {
  char *end;
  double dx = strtod(text, &end);

  if (end == text) {
    Serial.println("Enter two numbers, e.g. 5 -5");
    return;
  }

  char *second = end;
  double dy = strtod(second, &end);

  if (end == second) {
    Serial.println("Enter both X and Y.");
    return;
  }

  while (*end == ' ' || *end == '\t') end++;

  // Numerical guard, NOT a physical travel limit.
  if (*end != '\0' ||
      !isfinite(dx) || !isfinite(dy) ||
      fabs(dx) > 1000 || fabs(dy) > 1000) {
    Serial.println("Invalid movement.");
    return;
  }

  long aSteps = lround((dx + dy) * STEPS_PER_MM);
  long bSteps = lround((dx - dy) * STEPS_PER_MM);

  Serial.println("Moving...");

  if (moveCoreXY(aSteps, bSteps)) {
    Serial.println("Done.");
  }
  else {
    Serial.println("Stopped; movement incomplete.");
  }
}

// Called throughout a gantry movement.
// Servo commands are accepted; S stops the gantry.
// Other input is discarded until the move finishes.
bool stopRequested() {
  bool stop = false;

  while (Serial.available() > 0) {
    char c = Serial.read();

    if (handleServoCommand(c)) continue;

    if (c == 'S' || c == 's') {
      stop = true;
    }
  }

  return stop;
}

bool moveCoreXY(long aSteps, long bSteps) {
  digitalWrite(A_DIR, (aSteps >= 0) ^ INVERT_A);
  digitalWrite(B_DIR, (bSteps >= 0) ^ INVERT_B);
  delayMicroseconds(20);

  long aCount = labs(aSteps);
  long bCount = labs(bSteps);
  long ticks = max(aCount, bCount);

  if (ticks == 0) return true;

  long aError = 0;
  long bError = 0;
  unsigned long previousTick = micros();

  for (long i = 0; i < ticks; i++) {
    while ((unsigned long)(micros() - previousTick)
           < STEP_INTERVAL_US) {
      if (stopRequested()) return false;
    }

    if (stopRequested()) return false;
    previousTick = micros();

    aError += aCount;
    bError += bCount;

    if (aError >= ticks) {
      aError -= ticks;
      digitalWrite(A_PUL, LOW);
    }

    if (bError >= ticks) {
      bError -= ticks;
      digitalWrite(B_PUL, LOW);
    }

    delayMicroseconds(PULSE_WIDTH_US);

    digitalWrite(A_PUL, HIGH);
    digitalWrite(B_PUL, HIGH);
  }

  return true;
}


