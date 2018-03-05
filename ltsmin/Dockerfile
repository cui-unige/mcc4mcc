FROM alpine

COPY from-vm/local/bin/*            /usr/bin/
COPY from-vm/BenchKit/formulizer.py /usr/bin/

RUN true \
 && echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing" \
        >> /etc/apk/repositories \
 && apk add --no-cache --virtual .build-deps \
        build-base \
        curl \
        flex \
        unzip \
        apache-ant \
        perl \
        popt-dev \
        zlib-dev \
 && apk add --no-cache \
        bash \
        python3 \
        popt \
        zlib \
 && curl -L https://github.com/utwente-fmt/ltsmin/releases/download/2.1/ltsmin-2.1.tar.gz \
         --output ltsmin-2.1.tar.gz \
    && tar xf ltsmin-2.1.tar.gz \
    && cd ltsmin-2.1 \
    && sed -i".bak" "s/sysconf(_SC_LEVEL1_DCACHE_LINESIZE)/0/g" src/hre/hre_runtime.c \
    && ./configure --prefix=/usr \
    && make check \
    && make install \
    && cd .. \
    && rm -rf ltsmin-2.1* \
 && apk del .build-deps \
 && true

COPY mcc-head /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
