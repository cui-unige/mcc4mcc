FROM debian:stable-slim

COPY from-vm/ /tmp/from-vm

RUN true \
 && mv /tmp/from-vm/local/include/* /usr/local/include/ \
 && mv /tmp/from-vm/local/lib/*     /usr/local/lib/ \
 && mv /tmp/from-vm/local/GreatSPN  /usr/local/ \
 && mv /tmp/from-vm/MccCtlConv.jar  /usr/local/GreatSPN/bin/ \
 && rm -rf /tmp/from-vm \
 && mkdir -p /usr/share/man/man1/ \
 && apt-get update \
 && apt-get install \
            --yes \
            --fix-missing \
            bash \
            default-jre-headless \
            libgmpxx4ldbl \
            libgmp10 \
 && sed -i \
        -e 's/assistive_technologies/#assistive_technologies/g' \
        /etc/java-8-openjdk/accessibility.properties \
 && apt-get autoremove \
            --yes \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && true

COPY mcc-head /usr/local/bin/

VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head" ]
CMD        [ "--help" ]
