// Non-blocking transmit queue. See serial_link.h.
#include "serial_link.h"

#include <Arduino.h>

#include "protocol.h"

namespace serial_link {

namespace {

// Two maximum-length lines: enough for the longest burst the controller
// emits for one request (an ABORTED event followed by the STOP reply).
const uint8_t QUEUE_SIZE = 2 * proto::MAX_LINE_BYTES;

uint8_t queue[QUEUE_SIZE];
uint8_t head = 0;   // next byte to transmit
uint8_t count = 0;  // bytes queued

// Hands as many queued bytes to the driver as it can take without blocking.
void drainNonBlocking() {
  int room = Serial.availableForWrite();
  while (count > 0 && room > 0) {
    Serial.write(queue[head]);
    head = static_cast<uint8_t>(head + 1 == QUEUE_SIZE ? 0 : head + 1);
    --count;
    --room;
  }
}

}  // namespace

void begin(uint32_t baud) {
  head = 0;
  count = 0;
  Serial.begin(baud);
}

void enqueue(const char *line, uint8_t length) {
  for (uint8_t i = 0; i < length; ++i) {
    while (count >= QUEUE_SIZE) {
      // Queue full: wait for the driver (bounded by the baud rate, ~87 us/byte).
      drainNonBlocking();
    }
    uint16_t tail = static_cast<uint16_t>(head) + count;
    if (tail >= QUEUE_SIZE) tail -= QUEUE_SIZE;
    queue[tail] = static_cast<uint8_t>(line[i]);
    ++count;
  }
  drainNonBlocking();
}

void service() { drainNonBlocking(); }

}  // namespace serial_link
