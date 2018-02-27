FROM alpine

COPY from-vm/its-tools /usr/local/share/its-tools

RUN true \
 && apk add --no-cache --virtual .build-deps \
        autoconf \
        automake \
        build-base \
        curl \
        libtool \
        python3-dev \
 && apk add --no-cache \
        bash \
        libstdc++ \
        python3 \
 && curl http://www.lrde.epita.fr/dload/spot/spot-2.4.2.tar.gz \
         --output spot-2.4.2.tar.gz \
    && tar xf spot-2.4.2.tar.gz \
    && cd spot-2.4.2 \
    && sed -i".bak" "s/secure_getenv/getenv/g" spot/misc/tmpfile.cc \
    && ./configure \
    && make install \
    && cd .. \
    && rm -rf spot-2.4.2* \
 && apk del .build-deps \
 && true

COPY mcc-head /usr/local/bin

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
