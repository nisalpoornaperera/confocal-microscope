// Coordinated motion. See motion.h.
#include "motion.h"

#include <math.h>

namespace motion {

uint32_t tickIntervalQ8(const uint32_t distance[MOTORS], uint32_t ticks,
                        const float maxSpeed[MOTORS]) {
  // Duration of the move in seconds: the slowest motor relative to its limit.
  float duration = 0.0f;
  for (uint8_t i = 0; i < MOTORS; ++i) {
    const float t = static_cast<float>(distance[i]) / maxSpeed[i];
    if (t > duration) duration = t;
  }
  // Interval = duration / N, in 1/256 us. With max speeds >= 1 step/s it is
  // at most 1 s = 2.56e8 units, well inside 32 bits.
  const float q8 = duration * 256.0e6f / static_cast<float>(ticks);
  const float rounded = ceilf(q8);
  if (rounded < 1.0f) return 1u;
  if (rounded > 4.0e9f) return 4000000000u;
  return static_cast<uint32_t>(rounded);
}

// --------------------------------------------------------------------------- Interpolator

Interpolator::Interpolator() : total_(0), done_(0) {
  for (uint8_t i = 0; i < MOTORS; ++i) {
    distance_[i] = 0;
    error_[i] = 0;
    direction_[i] = 1;
  }
}

uint32_t Interpolator::begin(const int32_t start[MOTORS], const int32_t target[MOTORS]) {
  total_ = 0;
  done_ = 0;
  for (uint8_t i = 0; i < MOTORS; ++i) {
    const int32_t delta = target[i] - start[i];  // |delta| <= 2 * STEP_LIMIT
    direction_[i] = delta < 0 ? -1 : 1;
    distance_[i] = static_cast<uint32_t>(delta < 0 ? -delta : delta);
    if (distance_[i] > total_) total_ = distance_[i];
  }
  for (uint8_t i = 0; i < MOTORS; ++i) error_[i] = total_ / 2u;
  return total_;
}

uint8_t Interpolator::tick(int32_t position[MOTORS]) {
  if (done_ >= total_) return 0;
  uint8_t stepped = 0;
  for (uint8_t i = 0; i < MOTORS; ++i) {
    // error_ < N before the addition and distance_ <= N, so at most one step;
    // values stay below 2 * N <= 4e7, far inside 32 bits.
    error_[i] += distance_[i];
    if (error_[i] >= total_) {
      error_[i] -= total_;
      position[i] += direction_[i];
      stepped |= static_cast<uint8_t>(1u << i);
    }
  }
  ++done_;
  return stepped;
}

// --------------------------------------------------------------------------- TickClock

TickClock::TickClock() : dueUs_(0), intervalUs_(0), intervalFrac_(0), frac_(0) {}

void TickClock::addInterval() {
  const uint16_t frac = static_cast<uint16_t>(frac_) + intervalFrac_;
  dueUs_ += intervalUs_ + (frac >> 8);
  frac_ = static_cast<uint8_t>(frac & 0xFF);
}

void TickClock::start(uint32_t nowUs, uint32_t intervalQ8) {
  intervalUs_ = intervalQ8 >> 8;
  intervalFrac_ = static_cast<uint8_t>(intervalQ8 & 0xFF);
  frac_ = 0;
  dueUs_ = nowUs;
  addInterval();
}

bool TickClock::due(uint32_t nowUs) const {
  return static_cast<int32_t>(nowUs - dueUs_) >= 0;
}

void TickClock::advance(uint32_t nowUs) {
  addInterval();
  if (static_cast<int32_t>(nowUs - dueUs_) >= 0) {
    // Already late for the next tick: restart the schedule from now rather
    // than catching up with a burst of steps faster than the limit.
    dueUs_ = nowUs;
    frac_ = 0;
    addInterval();
  }
}

}  // namespace motion
