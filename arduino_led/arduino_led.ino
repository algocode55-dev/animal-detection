// Animal Detection Dashboard - Arduino LED Controller
// Built-in LED on Pin 13
// External LED on Pin 12 (connect positive long leg to 12, negative short leg to GND with a resistor)

const int BUILTIN_LED_PIN = 13;
const int EXTERNAL_LED_PIN = 12;

void setup() {
  // Initialize serial communication at 9600 bits per second
  Serial.begin(9600);
  
  // Set LED pins as output
  pinMode(EXTERNAL_LED_PIN, OUTPUT);
  
  // Start with LEDs off
  digitalWrite(EXTERNAL_LED_PIN, LOW);
}

void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming byte
    char incomingByte = Serial.read();
    
    // Command '1' means turn LEDs ON
    if (incomingByte == '1') {
      digitalWrite(EXTERNAL_LED_PIN, HIGH);
    }
    // Command '0' means turn LEDs OFF
    else if (incomingByte == '0') {
      digitalWrite(EXTERNAL_LED_PIN, LOW);
    }
  }
}
