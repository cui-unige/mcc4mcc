#! /usr/bin/env bash

case "${BK_TOOL}" in
  "mcc4mcc-full")
    python3 \
      -m mcc4mcc \
      --data="/usr/share/mcc4mcc" \
      run \
      --prefix="65b80f64"
  ;;
  "mcc4mcc-structural")
    python3 \
      -m mcc4mcc \
      --data="/usr/share/mcc4mcc" \
      run \
      --prefix="75f5f979"
  ;;
  "irma4mcc-full")
    python3 \
      -m mcc4mcc \
      --data="/usr/share/mcc4mcc" \
      run \
      --prefix="65b80f64" \
      --cheat
  ;;
  "irma4mcc-structural")
    python3 \
      -m mcc4mcc \
      --data="/usr/share/mcc4mcc" \
      run \
      --prefix="75f5f979" \
      --cheat
  ;;
esac
