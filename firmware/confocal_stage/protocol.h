// Serial line protocol of the motor controller (device side).
//
// Byte-for-byte compatible with backend/confocal/hardware/arduino/protocol.py,
// which is the specification (docs/serial-protocol.md describes it):
//
//   host -> device   <seq> <CMD> [args...]*<CRC>\n
//   device -> host   <seq> OK [payload...]*<CRC>\n
//                    <seq> ERR <CODE> <message>*<CRC>\n
//                    ! <EVENT> [payload...]*<CRC>\n
//
// <CRC> is CRC-8 (polynomial 0x07, init 0x00, no reflection, no final XOR)
// of every byte before the '*', as two hex digits (upper case when sending,
// either case accepted). Lines are at most MAX_LINE_BYTES bytes including the
// '\n' terminator.
//
// Hardware independent: compiled into the sketch and into the native unit
// tests (firmware/test/).
#ifndef CONFOCAL_PROTOCOL_H
#define CONFOCAL_PROTOCOL_H

#include <stdint.h>

namespace proto {

// Longest line in either direction, including the '\n' terminator.
const uint8_t MAX_LINE_BYTES = 96;
// Sequence number of a reply to a request whose own number was unreadable.
const uint16_t UNATTRIBUTED_SEQ = 0;
// Largest absolute step value carried by the protocol (7 digits).
const int32_t STEP_LIMIT = 9999999L;
// Longest firmware version string (keeps the STATUS reply within one line).
const uint8_t MAX_VERSION_LENGTH = 8;
// Number of motors (a, b, c).
const uint8_t MOTOR_COUNT = 3;

enum Command : uint8_t {
  CMD_MOVE = 0,  // MOVE <a> <b> <c>
  CMD_GETPOS,
  CMD_HOME,
  CMD_STOP,
  CMD_STATUS,
  CMD_PING,
  CMD_ZERO,
  CMD_RELEASE,
  CMD_COUNT
};

enum ErrorCode : uint8_t {
  E_NONE = 0,
  E_CRC,      // checksum missing or wrong
  E_SYNTAX,   // sequence number, characters or spacing
  E_LENGTH,   // line longer than MAX_LINE_BYTES
  E_UNKNOWN,  // unknown command word
  E_ARGS,     // wrong number or format of arguments
  E_BUSY,     // MOVE / HOME / ZERO / RELEASE while a move runs
  E_RANGE     // MOVE target outside a motor's travel
};

// One decoded host request. `seq` is also set when parsing fails: it is the
// number the ERR reply carries (UNATTRIBUTED_SEQ if the line's own number
// could not be read).
struct Request {
  uint16_t seq;
  Command command;
  uint8_t argCount;
  int32_t args[MOTOR_COUNT];
};

// CRC-8, polynomial 0x07, initial value 0x00 (crc8("123456789") == 0xF4).
uint8_t crc8(const uint8_t *data, uint8_t length);

// Number of integer arguments `command` takes.
uint8_t commandArity(Command command);

// Validates and decodes one received line. `line`/`length` exclude the '\n'
// terminator (a trailing '\r' is accepted and ignored). Returns E_NONE on
// success; otherwise the error code of the ERR reply, with `out.seq` set to
// the sequence number to attribute it to. Applies exactly the checks of
// protocol.py unframe() + parse_request(), in the same order.
ErrorCode parseRequest(const uint8_t *line, uint8_t length, Request &out);

// Assembles received bytes into lines without ever blocking or overflowing.
class LineReader {
 public:
  enum Result : uint8_t {
    PENDING = 0,  // no complete line yet
    LINE,         // a complete line is available: data() / length()
    TOO_LONG      // a line longer than MAX_LINE_BYTES ended (discarded)
  };

  LineReader();
  // Forgets any partial line.
  void reset();
  // Feeds one byte. After LINE, data()/length() stay valid until the next feed().
  Result feed(uint8_t byte);
  const uint8_t *data() const { return buffer_; }
  uint8_t length() const { return length_; }

 private:
  // Content bytes of the longest acceptable line (terminator excluded).
  static const uint8_t CAPACITY = MAX_LINE_BYTES - 1;
  uint8_t buffer_[CAPACITY];
  uint8_t length_;
  bool overflow_;
  bool complete_;
};

// Builds one framed outgoing line: body, '*', CRC-8 as two upper-case hex
// digits, '\n'. Appending past the line limit marks the builder as failed
// (finish() then returns 0) instead of writing out of bounds.
class LineBuilder {
 public:
  LineBuilder();
  void begin();
  void appendChar(char c);
  void appendText(const char *text);    // string in RAM
  void appendTextP(const char *text);   // string in flash (PSTR / PROGMEM)
  void appendSigned(int32_t value);
  void appendUnsigned(uint32_t value);
  // Appends "*XX\n"; returns the total length, or 0 if the line did not fit.
  uint8_t finish();
  const char *data() const { return buffer_; }

 private:
  static const uint8_t TRAILER = 4;  // "*XX\n"
  char buffer_[MAX_LINE_BYTES + 1];  // +1 keeps it NUL-terminated for debugging
  uint8_t length_;
  bool failed_;
};

// Appends "<CODE> <fixed message>" of an ERR reply (the firmware never
// echoes received bytes; the texts equal emulator.py _ERROR_TEXT).
void appendErrorText(LineBuilder &line, ErrorCode code);

}  // namespace proto

#endif  // CONFOCAL_PROTOCOL_H
