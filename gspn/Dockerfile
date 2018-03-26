FROM debian:stable-slim

COPY from-vm/local/include/*  /usr/include/
COPY from-vm/local/lib/*      /usr/lib/
COPY from-vm/local/GreatSPN/* /usr/share/GreatSPN/
COPY from-vm/MccCtlConv.jar   /usr/share/GreatSPN/

RUN true \
 && mkdir -p /usr/share/man/man1/ \
 && echo "APT::Install-Recommends "0";" >> /etc/apt/apt.conf.d/10norecommends \
 && echo "APT::Install-Suggests "0";"   >> /etc/apt/apt.conf.d/10norecommends \
 && apt-get update \
 && apt-get install \
            --yes \
            --fix-missing \
            bash \
            ca-certificates \
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

COPY mcc-head /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
