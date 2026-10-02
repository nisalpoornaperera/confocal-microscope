// Command dispatch and motion state machine. See controller.h.
#include "controller.h"

#include "pgm_compat.h"

namespace controller {

using proto::ErrorCode;
using proto::Request;

namespace {

// IN1..IN4 = bits 0..3: 1000 1100 0100 0110 0010 0011 0001 1001.
const uint8_t kHalfStep[8] PROGMEM = {0x01, 0x03, 0x02, 0x06, 0x04, 0x0C, 0x08, 0x09};

}  // namespace

uint8_t halfStepPattern(uint8_t phase) { return pgm_read_byte(&kHalfStep[phase & 0x07]); }

Controller::Controller()
    : limits_(0),
      version_(0),
      lineSink_(0),
      coilSink_(0),
      state_(STATE_IDLE),
      moving_(false),
      coilsEnabled_(false),
      moveSeq_(0) {
  for (uint8_t i = 0; i < MOTORS; ++i) {
    position_[i] = 0;
    phase_[i] = 0;
    maxSpeed_[i] = 1.0f;
  }
}

void Controller::begin(const MotorLimits *limits, const char *version, LineSink lineSink,
                       CoilSink coilSink) {
  limits_ = limits;
  version_ = version;
  lineSink_ = lineSink;
  coilSink_ = coilSink;
  for (uint8_t i = 0; i < MOTORS; ++i) {
    position_[i] = 0;
    phase_[i] = 0;
    maxSpeed_[i] = limits_[i].maxSpeedStepsPerS;
  }
  state_ = STATE_IDLE;
  moving_ = false;
  moveSeq_ = 0;
  releaseAll();

  out_.begin();
  out_.appendTextP(PSTR("! BOOT "));
  out_.appendText(version_);
  transmit();
}

// --------------------------------------------------------------------------- input

void Controller::handleLine(const uint8_t *line, uint8_t length, uint32_t nowUs) {
  poll(nowUs);  // a move whose last tick is due finishes before the command is seen
  Request request;
  const ErrorCode code = proto::parseRequest(line, length, request);
  if (code != proto::E_NONE) {
    sendError(request.seq, code);
    return;
  }
  dispatch(request, nowUs);
}

void Controller::handleOverlongLine(uint32_t nowUs) {
  poll(nowUs);
  sendError(proto::UNATTRIBUTED_SEQ, proto::E_LENGTH);
}

void Controller::poll(uint32_t nowUs) {
  if (!moving_ || !clock_.due(nowUs)) return;
  const uint8_t stepped = interpolator_.tick(position_);
  for (uint8_t i = 0; i < MOTORS; ++i) {
    if (stepped & (1u << i)) {
      phase_[i] = static_cast<uint8_t>((phase_[i] + interpolator_.direction(i)) & 0x07);
      coilSink_(i, halfStepPattern(phase_[i]));
    }
  }
  if (interpolator_.finished()) {
    moving_ = false;
    state_ = STATE_IDLE;
    sendMoveEvent(true);
  } else {
    clock_.advance(nowUs);
  }
}

// --------------------------------------------------------------------------- commands

void Controller::dispatch(const Request &request, uint32_t nowUs) {
  switch (request.command) {
    case proto::CMD_MOVE:
      startMove(request, request.args, STATE_MOVING, nowUs);
      break;
    case proto::CMD_HOME: {
      const int32_t origin[MOTORS] = {0, 0, 0};
      startMove(request, origin, STATE_HOMING, nowUs);
      break;
    }
    case proto::CMD_STOP:
      onStop(request);
      break;
    case proto::CMD_GETPOS:
      sendOkPosition(request.seq);
      break;
    case proto::CMD_STATUS:
      onStatus(request);
      break;
    case proto::CMD_PING:
      sendOk(request.seq);
      break;
    case proto::CMD_ZERO:
      if (moving_) {
        sendError(request.seq, proto::E_BUSY);
        return;
      }
      for (uint8_t i = 0; i < MOTORS; ++i) position_[i] = 0;  // phase_ is kept
      sendOkPosition(request.seq);
      break;
    case proto::CMD_RELEASE:
      if (moving_) {
        sendError(request.seq, proto::E_BUSY);
        return;
      }
      releaseAll();
      sendOk(request.seq);
      break;
    case proto::CMD_COUNT:
    default:
      sendError(request.seq, proto::E_UNKNOWN);
      break;
  }
}

void Controller::startMove(const Request &request, const int32_t target[MOTORS], State state,
                           uint32_t nowUs) {
  if (moving_) {
    sendError(request.seq, proto::E_BUSY);
    return;
  }
  for (uint8_t i = 0; i < MOTORS; ++i) {
    if (target[i] < limits_[i].minSteps || target[i] > limits_[i].maxSteps) {
      sendError(request.seq, proto::E_RANGE);
      return;
    }
  }
  energiseAll();
  moveSeq_ = request.seq;
  const uint32_t ticks = interpolator_.begin(position_, target);
  sendOk(request.seq);
  if (ticks == 0) {
    state_ = STATE_IDLE;
    sendMoveEvent(true);
    return;
  }
  clock_.start(nowUs, motion::tickIntervalQ8(interpolator_.distances(), ticks, maxSpeed_));
  moving_ = true;
  state_ = state;
}

void Controller::onStop(const Request &request) {
  if (moving_) {
    moving_ = false;
    state_ = STATE_STOPPED;  // coils stay energised and hold the position
    sendMoveEvent(false);
  }
  sendOkPosition(request.seq);
}

void Controller::onStatus(const Request &request) {
  beginReply(request.seq);
  out_.appendTextP(PSTR(" OK state="));
  switch (state_) {
    case STATE_MOVING: out_.appendTextP(PSTR("moving")); break;
    case STATE_HOMING: out_.appendTextP(PSTR("homing")); break;
    case STATE_STOPPED: out_.appendTextP(PSTR("stopped")); break;
    case STATE_IDLE:
    default: out_.appendTextP(PSTR("idle")); break;
  }
  static const char kKeys[] PROGMEM = "abc";
  for (uint8_t i = 0; i < MOTORS; ++i) {
    out_.appendChar(' ');
    out_.appendChar(static_cast<char>(pgm_read_byte(&kKeys[i])));
    out_.appendChar('=');
    out_.appendSigned(position_[i]);
  }
  out_.appendTextP(moving_ ? PSTR(" moving=1") : PSTR(" moving=0"));
  out_.appendTextP(coilsEnabled_ ? PSTR(" enabled=1") : PSTR(" enabled=0"));
  out_.appendTextP(PSTR(" version="));
  out_.appendText(version_);
  transmit();
}

// --------------------------------------------------------------------------- coils

void Controller::energiseAll() {
  // Re-applies the current phase: after RELEASE no step is lost or gained.
  for (uint8_t i = 0; i < MOTORS; ++i) coilSink_(i, halfStepPattern(phase_[i]));
  coilsEnabled_ = true;
}

void Controller::releaseAll() {
  for (uint8_t i = 0; i < MOTORS; ++i) coilSink_(i, 0);
  coilsEnabled_ = false;
}

// --------------------------------------------------------------------------- output

void Controller::beginReply(uint16_t seq) {
  out_.begin();
  out_.appendUnsigned(seq);
}

void Controller::appendPosition() {
  for (uint8_t i = 0; i < MOTORS; ++i) {
    out_.appendChar(' ');
    out_.appendSigned(position_[i]);
  }
}

void Controller::sendOk(uint16_t seq) {
  beginReply(seq);
  out_.appendTextP(PSTR(" OK"));
  transmit();
}

void Controller::sendOkPosition(uint16_t seq) {
  beginReply(seq);
  out_.appendTextP(PSTR(" OK"));
  appendPosition();
  transmit();
}

void Controller::sendError(uint16_t seq, ErrorCode code) {
  beginReply(seq);
  out_.appendTextP(PSTR(" ERR "));
  proto::appendErrorText(out_, code);
  transmit();
}

void Controller::sendMoveEvent(bool done) {
  out_.begin();
  out_.appendTextP(done ? PSTR("! DONE ") : PSTR("! ABORTED "));
  out_.appendUnsigned(moveSeq_);
  appendPosition();
  transmit();
}

void Controller::transmit() {
  const uint8_t length = out_.finish();
  if (length != 0 && lineSink_ != 0) lineSink_(out_.data(), length);
}

}  // namespace controller
