FROM alpine

RUN true \
 && apk add --no-cache --virtual .build-deps \
        autoconf \
        automake \
        bison \
        build-base \
        flex \
        gcc \
        g++ \
        git \
        libtool \
        gmp-dev \
 && apk add --no-cache \
        bash \
        libstdc++ \
        gmp \
 && git clone --depth=1 git://git.code.sf.net/p/meddly/code-git meddly \
    && cd meddly \
    && autoreconf -fi \
    && ./configure --prefix=/usr \
    && make install \
    && cd .. \
    && rm -rf meddly \
 && git clone --depth=1 https://github.com/asminer/smart.git \
    && cd smart \
    && sed -i -e 's|--prefix=$PWD|--prefix=/usr|g' Config-all \
    && ./Config-all \
    && cd bin-release \
    && make install \
    && cd ../.. \
    && rm -rf smart \
 && apk del .build-deps \
 && true

COPY mcc-head /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
