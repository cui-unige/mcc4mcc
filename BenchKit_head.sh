#! /usr/bin/env bash

case "${BK_TOOL}" in
  "mcc4mcc-full")
    python3 \
      -m mcc4mcc \
      --data="/usr/share/mcc4mcc" \
      run \
      --prefix="87bd5343c30a279e"
  ;;
  "mcc4mcc-structural")
    python3 \
      -m mcc4mcc \
      --data="/usr/share/mcc4mcc" \
      run \
      --prefix="7e556e9247727f60"
  ;;
  "irma4mcc-full")
    python3 \
      -m mcc4mcc \
      --data="/usr/share/mcc4mcc" \
      run \
      --prefix="87bd5343c30a279e" \
      --cheat
  ;;
  "irma4mcc-structural")
    python3 \
      -m mcc4mcc \
      --data="/usr/share/mcc4mcc" \
      run \
      --prefix="7e556e9247727f60" \
      --cheat
  ;;
esac
