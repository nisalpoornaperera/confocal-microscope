// Machine configuration of the motor controller firmware.
//
// The travel limits and speeds below are the firmware's last line of
// defence and MUST equal the [arduino.a], [arduino.b] and [arduino.c]
// sections of the Pi's configuration (config/confocal.pi.toml or
// /etc/confocal/confocal.toml; defaults in backend/confocal/config.py
// MotorConfig). If you change them there, change them here and re-flash.
//
// Pin mapping: docs/wiring.md §3. Do not change it without rewiring.
#ifndef CONFOCAL_CONFIG_H
#define CONFOCAL_CONFIG_H

#include <Arduino.h>

// Reported in the BOOT banner and STATUS reply: 1-8 characters of [0-9A-Za-z._+-].
#define CONFOCAL_FIRMWARE_VERSION "1.0.0"

// USB serial link to the Raspberry Pi (8N1, no flow control).
constexpr uint32_t SERIAL_BAUD = 115200UL;

// --- Travel limits (absolute firmware steps from the power-up origin) and
// --- maximum speeds (half-steps per second) of motors a, b, c.
// 28BYJ-48 in half-step mode: ~4076 steps per output revolution; 600 steps/s
// is reliable at 5 V, ~1000 is the practical maximum.
constexpr int32_t MOTOR_A_MIN_STEPS = -200000L;
constexpr int32_t MOTOR_A_MAX_STEPS = 200000L;
constexpr float MOTOR_A_MAX_SPEED = 600.0f;

constexpr int32_t MOTOR_B_MIN_STEPS = -200000L;
constexpr int32_t MOTOR_B_MAX_STEPS = 200000L;
constexpr float MOTOR_B_MAX_SPEED = 600.0f;

constexpr int32_t MOTOR_C_MIN_STEPS = -200000L;
constexpr int32_t MOTOR_C_MAX_STEPS = 200000L;
constexpr float MOTOR_C_MAX_SPEED = 600.0f;

// --- Coil pins: IN1..IN4 of each ULN2003 board (docs/wiring.md §3).
// D0/D1 are the USB serial link; D13 is avoided (the bootloader blinks it).
const uint8_t MOTOR_PINS[3][4] = {
    {2, 3, 4, 5},       // motor A: D2 D3 D4 D5
    {6, 7, 8, 9},       // motor B: D6 D7 D8 D9
    {A0, A1, A2, A3},   // motor C: A0 A1 A2 A3 (digital outputs)
};

// --- Hardware watchdog: resets the Uno if the main loop ever hangs for more
// than ~1 s. The Pi then sees an unexpected "! BOOT" and treats the position
// as lost (docs/serial-protocol.md §5). Set to 0 to disable (only needed for
// boards with an old, non-Optiboot bootloader that cannot recover from a
// watchdog reset).
#define CONFOCAL_USE_WATCHDOG 1

// ------------------------------------------------------------------ checks
namespace config_checks {
constexpr int32_t STEP_LIMIT = 9999999L;  // protocol.py STEP_LIMIT
}

static_assert(MOTOR_A_MIN_STEPS < 0 && 0 < MOTOR_A_MAX_STEPS, "motor A travel must include 0");
static_assert(MOTOR_B_MIN_STEPS < 0 && 0 < MOTOR_B_MAX_STEPS, "motor B travel must include 0");
static_assert(MOTOR_C_MIN_STEPS < 0 && 0 < MOTOR_C_MAX_STEPS, "motor C travel must include 0");
static_assert(MOTOR_A_MIN_STEPS >= -config_checks::STEP_LIMIT &&
                  MOTOR_A_MAX_STEPS <= config_checks::STEP_LIMIT,
              "motor A travel exceeds the protocol range +/-9999999");
static_assert(MOTOR_B_MIN_STEPS >= -config_checks::STEP_LIMIT &&
                  MOTOR_B_MAX_STEPS <= config_checks::STEP_LIMIT,
              "motor B travel exceeds the protocol range +/-9999999");
static_assert(MOTOR_C_MIN_STEPS >= -config_checks::STEP_LIMIT &&
                  MOTOR_C_MAX_STEPS <= config_checks::STEP_LIMIT,
              "motor C travel exceeds the protocol range +/-9999999");
// The Pi accepts 0 < max_speed_steps_s <= 2000; the firmware needs >= 1.
static_assert(MOTOR_A_MAX_SPEED >= 1.0f && MOTOR_A_MAX_SPEED <= 2000.0f, "motor A speed");
static_assert(MOTOR_B_MAX_SPEED >= 1.0f && MOTOR_B_MAX_SPEED <= 2000.0f, "motor B speed");
static_assert(MOTOR_C_MAX_SPEED >= 1.0f && MOTOR_C_MAX_SPEED <= 2000.0f, "motor C speed");
static_assert(sizeof(CONFOCAL_FIRMWARE_VERSION) >= 2 && sizeof(CONFOCAL_FIRMWARE_VERSION) <= 9,
              "firmware version must be 1-8 characters");

#endif  // CONFOCAL_CONFIG_H
