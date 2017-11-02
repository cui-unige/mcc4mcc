FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        autotools \
        build-base \
        clang \
        curl \
        flex \
        git \
        make \
 && apk add --no-cache \
        bash \
        libstdc++ \
        popt \
        zlib \
 && curl http://www.mcrl2.org/download/release/mcrl2-201707.1.tar.gz \
      --output=mcrl2-201707.1.tar.gz \
 && tar zxf mcrl2-201707.1.tar.gz \
 && cd mcrl2-201707.1 \
 &&
 && cd .. \
 && git clone https://github.com/utwente-fmt/ltsmin.git \
 && cd ltsmin \
 && autoreconf -vfi \
 && ./configure --prefix=/usr --disable-dependency-tracking \
 && make \
 && make check \
 && make install \
 && cd .. \
 && apk del .build-deps \
 && false

ENTRYPOINT ltsmin
CMD [ --help ]
