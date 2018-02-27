FROM debian:stable-slim

RUN true \
 && mkdir -p /usr/share/man/man1/ \
 && apt-get update \
 && apt-get install \
            --yes \
            --fix-missing \
            bash \
            default-jre-headless \
 && false

COPY mcc-head /usr/local/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
