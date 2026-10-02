// Confocal surface scanner - Delta Stage motor controller for the Arduino Uno.
//
// Drives motors a, b, c (28BYJ-48 via three ULN2003 boards, half-step) of
// the OpenFlexure Delta Stage and talks to the Raspberry Pi over USB serial
// (115200 8N1) with the CRC-framed line protocol of docs/serial-protocol.md.
// The executable specification is backend/confocal/hardware/arduino/emulator.py:
// for the same input lines this sketch sends the same bytes.
//
// Modules:
//   config.h            machine constants: travel limits, speeds, pins, version
//   protocol.h/.cpp     CRC-8, line assembly, request parsing, reply framing
//   motion.h/.cpp       multi-axis Bresenham interpolation + tick scheduler
//   controller.h/.cpp   command dispatch and motion state machine
//   coils.h/.cpp        ULN2003 coil outputs
//   serial_link.h/.cpp  non-blocking transmit queue
//
// The main loop never blocks: one interpolation tick is executed whenever it
// is due (micros()-based schedule), and serial input is read between ticks,
// so STOP, GETPOS, STATUS and PING are answered during a move and STOP takes
// effect before the next step.
//
// Build / flash: firmware/README.md (Arduino IDE 2, or firmware/build.py).
// Wiring and power: docs/wiring.md.
#include <Arduino.h>
#include <avr/wdt.h>

#include "coils.h"
#include "config.h"
#include "controller.h"
#include "protocol.h"
#include "serial_link.h"

// File-local state. `static` rather than an anonymous namespace: the Arduino
// IDE generates prototypes for the functions of a .ino at global scope.
static const controller::MotorLimits kLimits[3] = {
    {MOTOR_A_MIN_STEPS, MOTOR_A_MAX_STEPS, MOTOR_A_MAX_SPEED},
    {MOTOR_B_MIN_STEPS, MOTOR_B_MAX_STEPS, MOTOR_B_MAX_SPEED},
    {MOTOR_C_MIN_STEPS, MOTOR_C_MAX_STEPS, MOTOR_C_MAX_SPEED},
};

static const char kVersion[] = CONFOCAL_FIRMWARE_VERSION;

static controller::Controller stage;
static proto::LineReader reader;

// Reads the bytes that have arrived, up to and including one complete line,
// and hands that line to the controller.
static void serviceInput() {
  int available = Serial.available();
  while (available-- > 0) {
    const int byte = Serial.read();
    if (byte < 0) return;
    const proto::LineReader::Result result = reader.feed(static_cast<uint8_t>(byte));
    if (result == proto::LineReader::LINE) {
      stage.handleLine(reader.data(), reader.length(), micros());
      return;  // one line per pass: the next tick is checked before the next line
    }
    if (result == proto::LineReader::TOO_LONG) {
      stage.handleOverlongLine(micros());
      return;
    }
  }
}

void setup() {
  // A watchdog reset leaves the watchdog running: switch it off before anything else.
  MCUSR = 0;
  wdt_disable();

  coils::begin();  // all coil outputs LOW: motors de-energised
  serial_link::begin(SERIAL_BAUD);
  reader.reset();
  stage.begin(kLimits, kVersion, serial_link::enqueue, coils::apply);  // "! BOOT <version>"

#if CONFOCAL_USE_WATCHDOG
  wdt_enable(WDTO_1S);
#endif
}

void loop() {
#if CONFOCAL_USE_WATCHDOG
  wdt_reset();
#endif
  stage.poll(micros());
  serviceInput();
  serial_link::service();
}
