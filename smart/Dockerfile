FROM debian:stable-slim

RUN true \
 && echo "APT::Install-Recommends "0";" >> /etc/apt/apt.conf.d/10norecommends \
 && echo "APT::Install-Suggests "0";" >> /etc/apt/apt.conf.d/10norecommends \
 && apt-get update \
 && apt-get install \
            --yes \
            --fix-missing \
            bash \
            ca-certificates \
            default-jre-headless \
 && false

COPY mcc-head /usr/local/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
