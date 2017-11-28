FROM alpine

COPY mcc-head /usr/local/bin/

RUN true \
 && apk add --no-cache --virtual .build-deps \
        curl \
 && apk add --no-cache \
        bash \
 && curl http://www-dssz.informatik.tu-cottbus.de/track/download.php?id=200 \
         --output marcie.tar.gz \
    && tar xf marcie.tar.gz \
            -C /usr/local/ \
            --strip-components=1 \
    && rm marcie.tar.gz \
 && true

VOLUME [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head" ]
CMD [ "--help" ]
