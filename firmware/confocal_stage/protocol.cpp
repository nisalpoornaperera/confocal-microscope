// Serial line protocol (device side). See protocol.h.
#include "protocol.h"

#include "pgm_compat.h"

namespace proto {

namespace {

// Command words, indexed by Command. Kept in flash.
const char kMove[] PROGMEM = "MOVE";
const char kGetpos[] PROGMEM = "GETPOS";
const char kHome[] PROGMEM = "HOME";
const char kStop[] PROGMEM = "STOP";
const char kStatus[] PROGMEM = "STATUS";
const char kPing[] PROGMEM = "PING";
const char kZero[] PROGMEM = "ZERO";
const char kRelease[] PROGMEM = "RELEASE";

const char *const kCommandNames[CMD_COUNT] PROGMEM = {
    kMove, kGetpos, kHome, kStop, kStatus, kPing, kZero, kRelease,
};

// Most tokens a valid request has: <seq> <CMD> <a> <b> <c>.
const uint8_t MAX_TOKENS = 5;

#if defined(__AVR__)
const char *commandName(uint8_t index) {
  return reinterpret_cast<const char *>(pgm_read_word(&kCommandNames[index]));
}
#else
const char *commandName(uint8_t index) { return kCommandNames[index]; }
#endif

// True if the token equals the flash string `name`.
bool tokenEquals(const uint8_t *token, uint8_t length, const char *name) {
  for (uint8_t i = 0; i < length; ++i) {
    const char expected = static_cast<char>(pgm_read_byte(name + i));
    if (expected == '\0' || static_cast<uint8_t>(expected) != token[i]) return false;
  }
  return pgm_read_byte(name + length) == '\0';
}

bool isDigit(uint8_t c) { return c >= '0' && c <= '9'; }

// Value of one hex digit, or -1.
int8_t hexValue(uint8_t c) {
  if (c >= '0' && c <= '9') return static_cast<int8_t>(c - '0');
  if (c >= 'A' && c <= 'F') return static_cast<int8_t>(c - 'A' + 10);
  if (c >= 'a' && c <= 'f') return static_cast<int8_t>(c - 'a' + 10);
  return -1;
}

// protocol.py _SEQ_RE: [0-9]{1,5}, then 1..65535. Returns 0 if invalid.
uint16_t parseSeq(const uint8_t *token, uint8_t length) {
  if (length < 1 || length > 5) return 0;
  uint32_t value = 0;
  for (uint8_t i = 0; i < length; ++i) {
    if (!isDigit(token[i])) return 0;
    value = value * 10u + static_cast<uint32_t>(token[i] - '0');
  }
  if (value > 65535u) return 0;
  return static_cast<uint16_t>(value);
}

// protocol.py _INT_RE: -?[0-9]{1,7}.
bool parseStep(const uint8_t *token, uint8_t length, int32_t &out) {
  bool negative = false;
  if (length > 0 && token[0] == '-') {
    negative = true;
    ++token;
    --length;
  }
  if (length < 1 || length > 7) return false;
  int32_t value = 0;
  for (uint8_t i = 0; i < length; ++i) {
    if (!isDigit(token[i])) return false;
    value = value * 10 + static_cast<int32_t>(token[i] - '0');
  }
  out = negative ? -value : value;
  return true;
}

}  // namespace

uint8_t crc8(const uint8_t *data, uint8_t length) {
  uint8_t crc = 0x00;
  while (length--) {
    crc ^= *data++;
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x80) ? static_cast<uint8_t>((crc << 1) ^ 0x07)
                         : static_cast<uint8_t>(crc << 1);
    }
  }
  return crc;
}

uint8_t commandArity(Command command) { return command == CMD_MOVE ? 3 : 0; }

ErrorCode parseRequest(const uint8_t *line, uint8_t length, Request &out) {
  out.seq = UNATTRIBUTED_SEQ;
  out.command = CMD_PING;
  out.argCount = 0;
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i) out.args[i] = 0;

  // --- unframe(): length, checksum on the raw bytes, then characters.
  if (length > MAX_LINE_BYTES - 1) return E_LENGTH;  // + '\n' would exceed the limit
  if (length > 0 && line[length - 1] == '\r') --length;

  int16_t star = -1;
  for (uint8_t i = 0; i < length; ++i) {
    if (line[i] == '*') star = static_cast<int16_t>(i);
  }
  if (star < 0) return E_CRC;
  const uint8_t bodyLength = static_cast<uint8_t>(star);
  if (length - bodyLength - 1 != 2) return E_CRC;
  const int8_t high = hexValue(line[bodyLength + 1]);
  const int8_t low = hexValue(line[bodyLength + 2]);
  if (high < 0 || low < 0) return E_CRC;
  if (crc8(line, bodyLength) != static_cast<uint8_t>((high << 4) | low)) return E_CRC;

  if (bodyLength == 0) return E_SYNTAX;  // empty message
  for (uint8_t i = 0; i < bodyLength; ++i) {
    const uint8_t c = line[i];
    if (c < 0x20 || c > 0x7E || c == '*') return E_SYNTAX;  // non-ASCII / control / '*'
    if (c == ' ' && (i == 0 || i == bodyLength - 1 || line[i + 1] == ' ')) {
      return E_SYNTAX;  // leading, trailing or double space
    }
  }

  // --- parse_request(): tokens are now non-empty and single-space separated.
  uint8_t tokenStart[MAX_TOKENS];
  uint8_t tokenLength[MAX_TOKENS];
  uint8_t tokens = 0;
  uint8_t start = 0;
  for (uint8_t i = 0; i <= bodyLength; ++i) {
    if (i == bodyLength || line[i] == ' ') {
      if (tokens < MAX_TOKENS) {
        tokenStart[tokens] = start;
        tokenLength[tokens] = static_cast<uint8_t>(i - start);
      }
      if (tokens < 255) ++tokens;
      start = static_cast<uint8_t>(i + 1);
    }
  }

  const uint16_t seq = parseSeq(line + tokenStart[0], tokenLength[0]);
  if (seq == 0) return E_SYNTAX;  // unreadable or out of range: reply unattributed
  out.seq = seq;
  if (tokens < 2) return E_SYNTAX;

  uint8_t command = 0;
  while (command < CMD_COUNT &&
         !tokenEquals(line + tokenStart[1], tokenLength[1], commandName(command))) {
    ++command;
  }
  if (command == CMD_COUNT) return E_UNKNOWN;
  out.command = static_cast<Command>(command);

  const uint8_t arity = commandArity(out.command);
  if (tokens - 2 != arity) return E_ARGS;
  for (uint8_t i = 0; i < arity; ++i) {
    if (!parseStep(line + tokenStart[2 + i], tokenLength[2 + i], out.args[i])) return E_ARGS;
  }
  out.argCount = arity;
  return E_NONE;
}

// --------------------------------------------------------------------------- LineReader

LineReader::LineReader() { reset(); }

void LineReader::reset() {
  length_ = 0;
  overflow_ = false;
  complete_ = false;
}

LineReader::Result LineReader::feed(uint8_t byte) {
  if (complete_) {  // the previous line has been consumed by now
    length_ = 0;
    complete_ = false;
  }
  if (byte == '\n') {
    if (overflow_) {
      reset();
      return TOO_LONG;
    }
    complete_ = true;
    return LINE;
  }
  if (overflow_) return PENDING;  // discard up to the next '\n'
  if (length_ >= CAPACITY) {
    overflow_ = true;
    length_ = 0;
    return PENDING;
  }
  buffer_[length_++] = byte;
  return PENDING;
}

// --------------------------------------------------------------------------- LineBuilder

LineBuilder::LineBuilder() { begin(); }

void LineBuilder::begin() {
  length_ = 0;
  failed_ = false;
  buffer_[0] = '\0';
}

void LineBuilder::appendChar(char c) {
  if (failed_ || length_ + TRAILER >= MAX_LINE_BYTES) {  // keep room for the trailer
    failed_ = true;
    return;
  }
  buffer_[length_++] = c;
  buffer_[length_] = '\0';
}

void LineBuilder::appendText(const char *text) {
  while (*text != '\0') appendChar(*text++);
}

void LineBuilder::appendTextP(const char *text) {
  for (;;) {
    const char c = static_cast<char>(pgm_read_byte(text++));
    if (c == '\0') return;
    appendChar(c);
  }
}

void LineBuilder::appendUnsigned(uint32_t value) {
  char digits[10];
  uint8_t count = 0;
  do {
    digits[count++] = static_cast<char>('0' + value % 10u);
    value /= 10u;
  } while (value != 0);
  while (count > 0) appendChar(digits[--count]);
}

void LineBuilder::appendSigned(int32_t value) {
  if (value < 0) {
    appendChar('-');
    appendUnsigned(static_cast<uint32_t>(-(value + 1)) + 1u);  // safe for INT32_MIN
  } else {
    appendUnsigned(static_cast<uint32_t>(value));
  }
}

uint8_t LineBuilder::finish() {
  static const char kHex[] PROGMEM = "0123456789ABCDEF";
  if (failed_ || length_ == 0 || length_ + TRAILER > MAX_LINE_BYTES) return 0;
  const uint8_t crc = crc8(reinterpret_cast<const uint8_t *>(buffer_), length_);
  buffer_[length_++] = '*';
  buffer_[length_++] = static_cast<char>(pgm_read_byte(&kHex[crc >> 4]));
  buffer_[length_++] = static_cast<char>(pgm_read_byte(&kHex[crc & 0x0F]));
  buffer_[length_++] = '\n';
  buffer_[length_] = '\0';
  return length_;
}

void appendErrorText(LineBuilder &line, ErrorCode code) {
  switch (code) {
    case E_CRC: line.appendTextP(PSTR("E_CRC bad checksum")); break;
    case E_SYNTAX: line.appendTextP(PSTR("E_SYNTAX malformed line")); break;
    case E_LENGTH: line.appendTextP(PSTR("E_LENGTH line too long")); break;
    case E_UNKNOWN: line.appendTextP(PSTR("E_UNKNOWN unknown command")); break;
    case E_ARGS: line.appendTextP(PSTR("E_ARGS bad arguments")); break;
    case E_BUSY: line.appendTextP(PSTR("E_BUSY moving")); break;
    case E_RANGE: line.appendTextP(PSTR("E_RANGE outside travel")); break;
    case E_NONE:
    default: line.appendTextP(PSTR("E_SYNTAX malformed line")); break;
  }
}

}  // namespace proto
