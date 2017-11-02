FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        autoconf \
        automake \
        build-base \
        clang \
        git \
        libtool \
        python3-dev \
 && apk add --no-cache \
        bash \
        libstdc++ \
        python3 \
 && git clone https://gitlab.lrde.epita.fr/spot/spot.git \
 && cd spot \
 && autoreconf -fi \
 && ./configure \
 && make \
 && make install \
 && cd .. \
 && rm -rf spot \
 && apk del .build-deps \
 && true

ENTRYPOINT spot
CMD [ --help ]
