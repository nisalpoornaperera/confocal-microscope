// Coordinated, interpolated motion of the three motors (hardware independent).
//
// A move by d = (da, db, dc) steps is executed as N = max(|da|, |db|, |dc|)
// ticks of a multi-axis Bresenham interpolation (docs/serial-protocol.md §4,
// backend/confocal/hardware/arduino/emulator.py): every motor has an error
// accumulator that starts at N / 2 (integer division); each tick adds |d_i|
// and, when the accumulator reaches N, the motor takes one step towards its
// target and N is subtracted. After k ticks motor i has taken exactly
// floor((k * |d_i| + N / 2) / N) steps - the emulator's closed form - so no
// motor steps more than once per tick and all of them arrive at tick N.
//
// Tick timing: the move lasts T = max_i(|d_i| / max_speed_i) and ticks are
// T / N apart, so the motor with the furthest to go relative to its own speed
// limit runs at exactly its maximum rate and none exceeds its limit.
#ifndef CONFOCAL_MOTION_H
#define CONFOCAL_MOTION_H

#include <stdint.h>

#include "protocol.h"

namespace motion {

const uint8_t MOTORS = proto::MOTOR_COUNT;

// Interval between ticks in 1/256 microsecond units ("Q8"), rounded up so the
// step rate never exceeds any motor's limit. `distance[i]` = |d_i| (>= 0),
// `ticks` = N (> 0), `maxSpeed[i]` in steps/s (>= 1).
uint32_t tickIntervalQ8(const uint32_t distance[MOTORS], uint32_t ticks,
                        const float maxSpeed[MOTORS]);

// Bresenham interpolation of one move.
class Interpolator {
 public:
  Interpolator();
  // Plans a move from `start` to `target`; returns N (0: already there).
  uint32_t begin(const int32_t start[MOTORS], const int32_t target[MOTORS]);
  // Executes one tick: updates `position` and returns a bit mask of the motors
  // that stepped (bit i = motor i). Does nothing once finished().
  uint8_t tick(int32_t position[MOTORS]);
  // Direction of motor i in the current move: +1 or -1 (meaningful if it moves).
  int8_t direction(uint8_t motor) const { return direction_[motor]; }
  uint32_t distance(uint8_t motor) const { return distance_[motor]; }
  const uint32_t *distances() const { return distance_; }
  uint32_t ticksTotal() const { return total_; }
  uint32_t ticksDone() const { return done_; }
  bool finished() const { return done_ >= total_; }

 private:
  uint32_t distance_[MOTORS];
  uint32_t error_[MOTORS];
  int8_t direction_[MOTORS];
  uint32_t total_;
  uint32_t done_;
};

// Wrap-safe tick scheduler on a free-running 32-bit microsecond clock
// (Arduino micros(), wraps every ~71.6 minutes). Tick k of a move is due at
// start + k * interval; the fractional part of the interval is carried so
// long moves do not drift. If the caller falls more than a whole interval
// behind (a long blocking operation), the schedule restarts from "now"
// instead of bursting several steps at once, so the step rate can never
// exceed the limit.
class TickClock {
 public:
  TickClock();
  // First tick due one interval after `nowUs`.
  void start(uint32_t nowUs, uint32_t intervalQ8);
  bool due(uint32_t nowUs) const;
  // Call after executing a due tick: schedules the next one.
  void advance(uint32_t nowUs);
  uint32_t nextDueUs() const { return dueUs_; }

 private:
  void addInterval();
  uint32_t dueUs_;
  uint32_t intervalUs_;   // whole microseconds of the interval
  uint8_t intervalFrac_;  // 1/256 us
  uint8_t frac_;          // accumulated 1/256 us
};

}  // namespace motion

#endif  // CONFOCAL_MOTION_H
