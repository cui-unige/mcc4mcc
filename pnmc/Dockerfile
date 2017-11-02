FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        build-base \
        boost-dev \
        cmake \
        clang \
        git \
        make \
 && apk add --no-cache \
        bash \
        libstdc++ \
 && git clone https://github.com/ahamez/libsdd.git \
 && cd libsdd \
 && cmake . \
 && make \
 && make install \
 && cd .. \
 && rm -rf libsdd \
 && git clone https://github.com/USCiLab/cereal.git \
 && cd cereal \
 && cmake . \
 && make \
 && make install \
 && cd .. \
 && rm -rf cereal \
 && git clone https://github.com/ahamez/pnmc.git \
 && cd pnmc \
 && cmake . \
 && make \
 && make install \
 && cd .. \
 && rm -rf pnmc \
 && apk del .build-deps \
 && true

ENTRYPOINT pnmc
CMD [ --help ]
