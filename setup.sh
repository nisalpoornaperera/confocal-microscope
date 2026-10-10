#!/usr/bin/env bash
# Confocal surface scanner - one-click setup for the Raspberry Pi 5.
#
# Double-click this file in the File Manager and choose "Execute in Terminal",
# or run:   bash setup.sh            (options: bash setup.sh --help)
exec bash "$(dirname "${BASH_SOURCE[0]}")/deploy/install.sh" "$@"
