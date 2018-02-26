FROM debian:stable-slim

RUN true \
 && mkdir -p /usr/share/man/man1/ \
 && apt-get update \
 && apt-get install \
            --yes \
            --fix-missing \
            bash \
            curl \
            libgmpxx4ldbl \
            libgmp10 \
            default-jre-headless \
 && curl https://yanntm.github.io/ITS-commandline/itscl_linux.tgz \
         --output itscl_linux.tgz \
    && tar xf itscl_linux.tgz \
    && mv eclipse /usr/share/itscl \
    && rm -f itscl_linux.tgz \
    && rm -f runeclipse.sh \
 && apt-get purge \
            --yes \
            curl \
 && apt-get autoremove \
            --yes \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && true

COPY mcc-head /usr/local/bin/

VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head" ]
CMD        [ "--help" ]
