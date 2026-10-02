// Native (host PC) tests of the firmware's hardware-independent core.
//
// Replays firmware/test/vectors.txt - recorded from the Python reference
// (protocol.py / emulator.py) by gen_vectors.py - against protocol.cpp,
// motion.cpp and controller.cpp compiled for the host, and checks:
//   * CRC-8 values;
//   * Bresenham positions after k ticks (the emulator's closed form);
//   * tick intervals (never faster than the speed limit, never noticeably slower);
//   * complete serial sessions: every emitted byte equals the emulator's, with
//     input fed byte by byte through LineReader exactly as the sketch's loop()
//     does and simulated time advancing in POLL_US steps;
//   * coil outputs: every step changes exactly the stepping motor's coils to the
//     adjacent half-step phase (no lost / extra / jumping steps), motors are
//     energised during moves and released by RELEASE / BOOT;
//   * step timing: no motor ever steps faster than its configured limit.
//
// Build and run: python firmware/test/run_tests.py (see its docstring).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "controller.h"
#include "motion.h"
#include "protocol.h"

namespace {

const uint32_t POLL_US = 20;  // gen_vectors.py POLL_US

int g_failures = 0;
int g_checks = 0;
std::string g_context;

void fail(const std::string &message) {
  ++g_failures;
  if (g_failures <= 30) std::fprintf(stderr, "FAIL [%s] %s\n", g_context.c_str(), message.c_str());
}

void check(bool ok, const std::string &message) {
  ++g_checks;
  if (!ok) fail(message);
}

std::vector<uint8_t> fromHex(const std::string &hex) {
  std::vector<uint8_t> out;
  if (hex == "-") return out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back(static_cast<uint8_t>(std::strtoul(hex.substr(i, 2).c_str(), 0, 16)));
  }
  return out;
}

std::string printable(const std::vector<uint8_t> &bytes) {
  std::string s;
  char buf[8];
  for (uint8_t b : bytes) {
    if (b >= 0x20 && b < 0x7F && b != '\\') {
      s += static_cast<char>(b);
    } else {
      std::snprintf(buf, sizeof buf, "\\x%02X", b);
      s += buf;
    }
  }
  return s;
}

// ---------------------------------------------------------------- simulated hardware

struct Hardware {
  std::vector<std::vector<uint8_t>> lines;  // emitted lines
  uint8_t pattern[3];                       // current coil pattern of each motor
  std::vector<std::pair<uint8_t, uint8_t>> coilCalls;  // (motor, pattern) since last reset
} g_hw;

void sinkLine(const char *line, uint8_t length) {
  g_hw.lines.push_back(std::vector<uint8_t>(line, line + length));
}

void sinkCoil(uint8_t motor, uint8_t pattern) {
  g_hw.pattern[motor] = pattern;
  g_hw.coilCalls.push_back(std::make_pair(motor, pattern));
}

int phaseOf(uint8_t pattern) {
  for (int p = 0; p < 8; ++p) {
    if (controller::halfStepPattern(static_cast<uint8_t>(p)) == pattern) return p;
  }
  return -1;
}

// ---------------------------------------------------------------- unit vectors

void testHalfStepTable() {
  g_context = "half-step table";
  // docs/serial-protocol.md §6: IN1..IN4 per phase.
  const char *rows[8] = {"1000", "1100", "0100", "0110", "0010", "0011", "0001", "1001"};
  for (int p = 0; p < 8; ++p) {
    uint8_t expected = 0;
    for (int coil = 0; coil < 4; ++coil) {
      if (rows[p][coil] == '1') expected |= static_cast<uint8_t>(1u << coil);
    }
    check(controller::halfStepPattern(static_cast<uint8_t>(p)) == expected, "phase pattern");
  }
}

void testCrc(const std::string &dataHex, const std::string &crcHex) {
  g_context = "CRC " + dataHex.substr(0, 16);
  std::vector<uint8_t> data = fromHex(dataHex);
  const unsigned expected = static_cast<unsigned>(std::strtoul(crcHex.c_str(), 0, 16));
  const unsigned actual = proto::crc8(data.empty() ? 0 : &data[0], static_cast<uint8_t>(data.size()));
  check(actual == expected, "crc8 mismatch");
}

struct BresGroup {
  int32_t start[3];
  int32_t target[3];
  motion::Interpolator interp;
  int32_t position[3];
  bool active;
} g_bres = {};

void testBresenham(std::istringstream &in) {
  int32_t start[3], target[3], expected[3];
  long long k;
  in >> start[0] >> start[1] >> start[2] >> target[0] >> target[1] >> target[2] >> k >>
      expected[0] >> expected[1] >> expected[2];
  g_context = "BRES";
  bool same = g_bres.active;
  for (int i = 0; i < 3 && same; ++i) {
    same = g_bres.start[i] == start[i] && g_bres.target[i] == target[i] &&
           static_cast<long long>(g_bres.interp.ticksDone()) <= k;
  }
  if (!same) {
    for (int i = 0; i < 3; ++i) {
      g_bres.start[i] = start[i];
      g_bres.target[i] = target[i];
      g_bres.position[i] = start[i];
    }
    g_bres.interp.begin(start, target);
    g_bres.active = true;
  }
  while (static_cast<long long>(g_bres.interp.ticksDone()) < k) {
    const uint8_t stepped = g_bres.interp.tick(g_bres.position);
    (void)stepped;
  }
  for (int i = 0; i < 3; ++i) check(g_bres.position[i] == expected[i], "Bresenham position");
  if (static_cast<long long>(g_bres.interp.ticksTotal()) == k) {
    check(g_bres.interp.finished(), "finished after N ticks");
    for (int i = 0; i < 3; ++i) check(g_bres.position[i] == target[i], "arrives at target");
  }
}

void testPeriod(std::istringstream &in) {
  uint32_t distance[3];
  float speed[3];
  double ideal;
  in >> distance[0] >> distance[1] >> distance[2] >> speed[0] >> speed[1] >> speed[2] >> ideal;
  g_context = "PERIOD";
  uint32_t n = distance[0];
  for (int i = 1; i < 3; ++i) n = distance[i] > n ? distance[i] : n;
  const double actual = motion::tickIntervalQ8(distance, n, speed) / 256.0;
  // Never shorter than ideal (beyond float precision), never more than 1/256 us longer.
  check(actual >= ideal * (1.0 - 1e-6), "interval shorter than the speed limit allows");
  check(actual <= ideal * (1.0 + 1e-6) + 1.0 / 256.0 + 1e-9, "interval needlessly long");
}

void testTickClockWrap() {
  g_context = "TickClock wrap";
  motion::TickClock clock;
  const uint32_t start = 0xFFFFFF00u;  // micros() about to wrap
  clock.start(start, 1000u * 256u);    // 1000 us
  check(!clock.due(start + 999u), "not due early across wrap");
  check(clock.due(start + 1000u), "due across wrap");
  clock.advance(start + 1000u);
  check(clock.nextDueUs() == start + 2000u, "next due across wrap");
  // Late by more than an interval: restart from now, no burst.
  clock.advance(start + 10000u);
  check(clock.nextDueUs() == start + 11000u, "re-anchor after a stall");
  // Fractional interval carries: 1000.5 us -> 2 ticks = 2001 us.
  clock.start(0, 1000u * 256u + 128u);
  clock.advance(1000u);
  clock.advance(2000u);
  check(clock.nextDueUs() == 3001u, "fraction carried");
}

void testLineReader() {
  g_context = "LineReader";
  proto::LineReader reader;
  const char *text = "12 PING*00\r\nX\n";
  std::vector<int> results;
  for (const char *p = text; *p; ++p) results.push_back(reader.feed(static_cast<uint8_t>(*p)));
  check(results[11] == proto::LineReader::LINE, "first line complete");
  check(results[13] == proto::LineReader::LINE && reader.length() == 1 && reader.data()[0] == 'X',
        "second line after the first");
  for (int i = 0; i < 95; ++i) reader.feed('a');
  check(reader.feed('\n') == proto::LineReader::LINE && reader.length() == 95, "95 bytes fit");
  for (int i = 0; i < 96; ++i) reader.feed('a');
  check(reader.feed('\n') == proto::LineReader::TOO_LONG, "96 bytes + newline too long");
  check(reader.feed('\n') == proto::LineReader::LINE && reader.length() == 0, "resynchronised");
}

// ---------------------------------------------------------------- scenarios

struct Scenario {
  std::string name;
  controller::MotorLimits limits[3];
  controller::Controller ctl;
  proto::LineReader reader;
  uint32_t now;
  std::vector<std::vector<uint8_t>> pending;  // actual output not yet compared
  size_t pendingIndex;
  // Step timing check.
  bool haveStep[3];
  uint32_t lastStepUs[3];
  double fastestSpeed;
};

void compareRemaining(Scenario &s) {
  for (size_t i = s.pendingIndex; i < s.pending.size(); ++i) {
    fail("unexpected extra output: " + printable(s.pending[i]));
  }
  s.pending.clear();
  s.pendingIndex = 0;
}

void collect(Scenario &s) {
  for (size_t i = 0; i < g_hw.lines.size(); ++i) s.pending.push_back(g_hw.lines[i]);
  g_hw.lines.clear();
}

// One loop() pass worth of polling, with coil and timing checks.
void pollChecked(Scenario &s) {
  int32_t before[3];
  for (int i = 0; i < 3; ++i) before[i] = s.ctl.position(i);
  uint8_t patternBefore[3];
  for (int i = 0; i < 3; ++i) patternBefore[i] = g_hw.pattern[i];
  g_hw.coilCalls.clear();
  s.ctl.poll(s.now);
  int calls[3] = {0, 0, 0};
  for (size_t c = 0; c < g_hw.coilCalls.size(); ++c) ++calls[g_hw.coilCalls[c].first];
  for (int i = 0; i < 3; ++i) {
    const int32_t delta = s.ctl.position(i) - before[i];
    if (delta == 0) {
      check(calls[i] == 0, "coils changed without a step");
      continue;
    }
    check(delta == 1 || delta == -1, "more than one step per tick");
    check(calls[i] == 1, "one coil update per step");
    const int p0 = phaseOf(patternBefore[i]);
    const int p1 = phaseOf(g_hw.pattern[i]);
    check(p0 >= 0 && p1 >= 0, "coils energised while stepping");
    check(((p0 + delta) & 7) == p1, "step goes to the adjacent half-step phase");
    check(p1 == s.ctl.phase(i), "phase bookkeeping");
    if (s.haveStep[i]) {
      // A motor steps at most once per tick, and ticks are never closer than
      // 1 / (fastest speed limit). (With equal limits - the normal setup - that
      // is each motor's own limit; with unequal limits a non-dominant motor's
      // *average* rate is within its limit, as in the emulator.) A tick can run
      // up to one poll late, so the following one may come up to POLL_US early.
      const double minInterval = 1e6 / s.fastestSpeed;
      const uint32_t interval = s.now - s.lastStepUs[i];
      check(interval + POLL_US + 1 >= minInterval, "motor stepped faster than the tick rate");
    }
    s.haveStep[i] = true;
    s.lastStepUs[i] = s.now;
  }
}

// Lets `dt` microseconds pass. loop() polls continuously; to keep long moves
// fast to simulate, time jumps straight to the next due tick plus a
// deterministic 0..POLL_US-1 us lateness (the real loop's jitter), and to the
// end when nothing moves. Polls between those instants would do nothing.
void advance(Scenario &s, uint32_t dt) {
  const uint32_t end = s.now + dt;
  while (s.now != end) {
    uint32_t next = end;
    if (s.ctl.moving()) {
      const uint32_t due = s.ctl.nextTickDueUs();
      const uint32_t late = (due * 2654435761u >> 16) % POLL_US;
      next = due + late;
      if (static_cast<int32_t>(next - s.now) <= 0) next = s.now + 1;
      if (static_cast<int32_t>(next - end) > 0) next = end;
    }
    s.now = next;
    pollChecked(s);
  }
}

void runScenario(std::vector<std::string> &records) {
  Scenario *s = new Scenario();
  std::istringstream header(records[0]);
  std::string word;
  header >> word >> s->name;
  {
    std::istringstream lim(records[1]);
    lim >> word;
    for (int i = 0; i < 3; ++i) {
      long lo, hi;
      double speed;
      lim >> lo >> hi >> speed;
      s->limits[i].minSteps = lo;
      s->limits[i].maxSteps = hi;
      s->limits[i].maxSpeedStepsPerS = static_cast<float>(speed);
      if (speed > s->fastestSpeed) s->fastestSpeed = speed;
    }
  }
  static const char kVersion[] = "1.0.0";  // emulator.FIRMWARE_VERSION
  s->now = 0x7FFFF000u;  // start near a micros() sign boundary on purpose
  g_hw.lines.clear();
  std::memset(g_hw.pattern, 0xEE, sizeof g_hw.pattern);

  for (size_t r = 2; r < records.size(); ++r) {
    std::istringstream in(records[r]);
    std::string kind;
    in >> kind;
    std::ostringstream ctx;
    ctx << s->name << " record " << r << " (" << records[r].substr(0, 40) << ")";
    if (kind == "OUT") {
      std::string hex;
      in >> hex;
      const std::vector<uint8_t> expected = fromHex(hex);
      if (s->pendingIndex >= s->pending.size()) {
        fail("missing output, expected: " + printable(expected));
      } else {
        const std::vector<uint8_t> &actual = s->pending[s->pendingIndex++];
        ++g_checks;
        if (actual != expected) {
          fail("output differs\n  expected: " + printable(expected) + "\n  actual:   " + printable(actual));
        }
      }
      continue;
    }
    compareRemaining(*s);
    g_context = ctx.str();
    if (kind == "BOOT") {
      s->ctl.begin(s->limits, kVersion, sinkLine, sinkCoil);
      s->reader.reset();
      for (int i = 0; i < 3; ++i) {
        check(g_hw.pattern[i] == 0, "BOOT de-energises every coil");
        s->haveStep[i] = false;
      }
      check(!s->ctl.coilsEnabled(), "BOOT: coils flag off");
      collect(*s);
    } else if (kind == "IN") {
      std::string hex;
      in >> hex;
      const std::vector<uint8_t> bytes = fromHex(hex);
      const bool wasEnabled = s->ctl.coilsEnabled();
      for (size_t i = 0; i < bytes.size(); ++i) {
        const proto::LineReader::Result result = s->reader.feed(bytes[i]);
        if (result == proto::LineReader::LINE) {
          s->ctl.handleLine(s->reader.data(), s->reader.length(), s->now);
        } else if (result == proto::LineReader::TOO_LONG) {
          s->ctl.handleOverlongLine(s->now);
        }
      }
      // Coil flag and physical outputs agree.
      for (int i = 0; i < 3; ++i) {
        if (s->ctl.coilsEnabled()) {
          check(phaseOf(g_hw.pattern[i]) == s->ctl.phase(i), "energised coils show the phase");
        } else {
          check(g_hw.pattern[i] == 0, "released coils are off");
        }
      }
      (void)wasEnabled;
      collect(*s);
    } else if (kind == "ADV") {
      unsigned long long dt;
      in >> dt;
      // Long waits in chunks: the microsecond clock wraps every 2^32 us (~71.6 min).
      while (dt > 0) {
        const uint32_t chunk = static_cast<uint32_t>(dt > 0x40000000ull ? 0x40000000ull : dt);
        advance(*s, chunk);
        dt -= chunk;
      }
      collect(*s);
    } else if (kind == "END") {
      break;
    } else {
      fail("unknown record " + kind);
    }
  }
  compareRemaining(*s);
  delete s;
}

}  // namespace

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "vectors.txt";
  std::ifstream file(path);
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return 2;
  }
  testHalfStepTable();
  testTickClockWrap();
  testLineReader();

  int crc = 0, bres = 0, period = 0, scenarios = 0;
  std::string line;
  std::vector<std::string> scenario;
  while (std::getline(file, line)) {
    if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
    if (line.empty() || line[0] == '#') continue;
    if (!scenario.empty()) {
      scenario.push_back(line);
      if (line == "END") {
        runScenario(scenario);
        scenario.clear();
        ++scenarios;
      }
      continue;
    }
    std::istringstream in(line);
    std::string kind;
    in >> kind;
    if (kind == "CRC") {
      std::string data, value;
      in >> data >> value;
      testCrc(data, value);
      ++crc;
    } else if (kind == "BRES") {
      testBresenham(in);
      ++bres;
    } else if (kind == "PERIOD") {
      testPeriod(in);
      ++period;
    } else if (kind == "SCENARIO") {
      scenario.push_back(line);
    } else {
      g_context = "parser";
      fail("unknown record: " + line);
    }
  }
  if (!scenario.empty()) fail("unterminated scenario");
  std::printf("vectors: %d CRC, %d Bresenham, %d interval, %d scenarios; %d checks, %d failures\n",
              crc, bres, period, scenarios, g_checks, g_failures);
  if (g_failures == 0 && scenarios == 0) {
    std::fprintf(stderr, "no scenarios found in %s\n", path);
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
