FROM alpine

RUN true \
 && apk add --no-cache --virtual .build-deps \
        curl \
 && apk add --no-cache \
        bash \
 # && curl http://www-dssz.informatik.tu-cottbus.de/track/download.php?id=200 \
 #         --output marcie.tar.gz \
 #    && tar xf marcie.tar.gz \
 #            -C /usr/ \
 #            --strip-components=1 \
 #    && rm marcie.tar.gz \
 && apk del .build-deps \
 && true

# Cannot use the real marcie tool, because its output does not conform
# to the expected one for the MCC.
COPY from-vm/marcie /usr/bin/
COPY mcc-head       /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
