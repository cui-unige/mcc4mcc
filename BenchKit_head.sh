#! /usr/bin/env bash

if [ ! -z "${BK_CHEAT+x}" ]
then
  cheat="--cheat"
fi

python3 \
  -m mcc4mcc \
  --data="/usr/share/mcc4mcc" \
  run \
  ${cheat}
