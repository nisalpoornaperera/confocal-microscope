// Portability shim for strings kept in flash (PROGMEM).
//
// On the AVR (Arduino Uno) constant strings are read from flash with the
// *_P functions of <avr/pgmspace.h> so they do not occupy any of the Uno's
// 2 KB of RAM. On a host PC (native unit tests, firmware/test/) there is no
// separate flash address space and the same names map to the plain C
// functions. Only the hardware-independent modules (protocol, motion,
// controller) include this header.
#ifndef CONFOCAL_PGM_COMPAT_H
#define CONFOCAL_PGM_COMPAT_H

#if defined(__AVR__)
#include <avr/pgmspace.h>
#else
#include <string.h>
#ifndef PROGMEM
#define PROGMEM
#endif
#ifndef PSTR
#define PSTR(s) (s)
#endif
#ifndef pgm_read_byte
#define pgm_read_byte(addr) (*(const unsigned char *)(addr))
#endif
#endif

#endif  // CONFOCAL_PGM_COMPAT_H
