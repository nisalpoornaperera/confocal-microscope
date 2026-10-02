// Non-blocking transmit queue in front of the hardware serial port (Arduino only).
//
// HardwareSerial has a 64-byte transmit buffer and Serial.write() blocks when
// it is full; at 115200 baud a 92-byte STATUS reply would stall the main loop
// for ~3 ms, longer than one step period at 600 steps/s. Lines are therefore
// queued here and handed to the serial driver only as fast as it has room
// (Serial.availableForWrite()), so stepping keeps its timing while replies are
// sent. Only if the queue itself is full (a host flooding the link with
// requests) does enqueue() wait for space - lines are never dropped or
// reordered.
#ifndef CONFOCAL_SERIAL_LINK_H
#define CONFOCAL_SERIAL_LINK_H

#include <stdint.h>

namespace serial_link {

// Opens the serial port.
void begin(uint32_t baud);

// Queues one complete line for transmission. Matches controller::LineSink.
void enqueue(const char *line, uint8_t length);

// Moves queued bytes into the serial driver without blocking. Call every loop.
void service();

}  // namespace serial_link

#endif  // CONFOCAL_SERIAL_LINK_H
