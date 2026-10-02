// Coil outputs of the three ULN2003 boards. See coils.h.
#include "coils.h"

#include <Arduino.h>

#include "config.h"

namespace coils {

void begin() {
  for (uint8_t motor = 0; motor < 3; ++motor) {
    for (uint8_t coil = 0; coil < 4; ++coil) {
      const uint8_t pin = MOTOR_PINS[motor][coil];
      digitalWrite(pin, LOW);  // output latch low before the pin becomes an output
      pinMode(pin, OUTPUT);
      digitalWrite(pin, LOW);
    }
  }
}

void apply(uint8_t motor, uint8_t pattern) {
  if (motor >= 3) return;
  for (uint8_t coil = 0; coil < 4; ++coil) {
    digitalWrite(MOTOR_PINS[motor][coil], (pattern >> coil) & 0x01 ? HIGH : LOW);
  }
}

}  // namespace coils
