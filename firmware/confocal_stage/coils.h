// Coil outputs of the three ULN2003 boards (Arduino only).
#ifndef CONFOCAL_COILS_H
#define CONFOCAL_COILS_H

#include <stdint.h>

namespace coils {

// Drives every coil pin LOW, then makes it an output: the motors are
// de-energised from the first instant (call first thing in setup()).
void begin();

// Sets IN1..IN4 of `motor` (0 = A, 1 = B, 2 = C) to bits 0..3 of `pattern`;
// 0 de-energises the motor. Matches controller::CoilSink.
void apply(uint8_t motor, uint8_t pattern);

}  // namespace coils

#endif  // CONFOCAL_COILS_H
