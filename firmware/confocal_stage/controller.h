// Command dispatch and motion state machine of the firmware (hardware independent).
//
// For every sequence of received lines and elapsed time, Controller emits
// the same bytes as backend/confocal/hardware/arduino/emulator.py
// (FirmwareEmulator), which is the executable specification. The sketch
// connects it to the Uno's serial port and coil pins; the native tests in
// firmware/test/ connect it to recorded emulator output.
//
// Time is passed in explicitly (a free-running microsecond clock such as
// Arduino micros()); nothing here blocks or waits.
#ifndef CONFOCAL_CONTROLLER_H
#define CONFOCAL_CONTROLLER_H

#include <stdint.h>

#include "motion.h"
#include "protocol.h"

namespace controller {

const uint8_t MOTORS = proto::MOTOR_COUNT;

// Firmware travel and speed limit of one motor; must equal [arduino.a|b|c]
// of the Pi's configuration. min_steps < 0 < max_steps.
struct MotorLimits {
  int32_t minSteps;
  int32_t maxSteps;
  float maxSpeedStepsPerS;
};

// STATUS "state" field.
enum State : uint8_t { STATE_IDLE = 0, STATE_MOVING, STATE_HOMING, STATE_STOPPED };

// Receives one complete framed line (terminator included) to transmit.
typedef void (*LineSink)(const char *line, uint8_t length);
// Drives the four coil inputs (IN1..IN4 = bits 0..3) of one motor; 0 = off.
typedef void (*CoilSink)(uint8_t motor, uint8_t pattern);

// 28BYJ-48 half-step sequence (docs/serial-protocol.md §6), bit 0 = IN1.
uint8_t halfStepPattern(uint8_t phase);

class Controller {
 public:
  Controller();

  // Power-up / reset: counters 0, coils off, state idle, emits "! BOOT <version>".
  // `limits` and `version` (1-8 characters of [0-9A-Za-z._+-], in RAM) must
  // outlive the controller.
  void begin(const MotorLimits *limits, const char *version, LineSink lineSink,
             CoilSink coilSink);

  // One received line without its '\n' terminator (from LineReader::LINE).
  void handleLine(const uint8_t *line, uint8_t length, uint32_t nowUs);
  // A line longer than MAX_LINE_BYTES ended (LineReader::TOO_LONG).
  void handleOverlongLine(uint32_t nowUs);
  // Runs the next interpolation tick if it is due; emits "! DONE" at the end.
  void poll(uint32_t nowUs);

  // Observation (tests / diagnostics).
  int32_t position(uint8_t motor) const { return position_[motor]; }
  State state() const { return state_; }
  bool moving() const { return moving_; }
  bool coilsEnabled() const { return coilsEnabled_; }
  uint8_t phase(uint8_t motor) const { return phase_[motor]; }
  // When the next interpolation tick is due (meaningful while moving()).
  uint32_t nextTickDueUs() const { return clock_.nextDueUs(); }

 private:
  void dispatch(const proto::Request &request, uint32_t nowUs);
  void startMove(const proto::Request &request, const int32_t target[MOTORS], State state,
                 uint32_t nowUs);
  void onStop(const proto::Request &request);
  void onStatus(const proto::Request &request);

  void energiseAll();
  void releaseAll();

  void sendOk(uint16_t seq);
  void sendOkPosition(uint16_t seq);
  void sendError(uint16_t seq, proto::ErrorCode code);
  void sendMoveEvent(bool done);
  void beginReply(uint16_t seq);
  void appendPosition();
  void transmit();

  const MotorLimits *limits_;
  const char *version_;
  LineSink lineSink_;
  CoilSink coilSink_;

  int32_t position_[MOTORS];
  // Electrical phase (0..7) of each motor. Kept separately from the counter
  // so ZERO (which resets the counters) never makes a motor jump phase.
  uint8_t phase_[MOTORS];
  State state_;
  bool moving_;
  bool coilsEnabled_;
  uint16_t moveSeq_;
  float maxSpeed_[MOTORS];

  motion::Interpolator interpolator_;
  motion::TickClock clock_;
  proto::LineBuilder out_;
};

}  // namespace controller

#endif  // CONFOCAL_CONTROLLER_H
