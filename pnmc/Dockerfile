FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        build-base \
        boost-dev \
        cmake \
        git \
 && apk add --no-cache \
        bash \
        libstdc++ \
 && git clone https://github.com/ahamez/libsdd.git \
<<<<<<< HEAD
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
=======
 && cd libsdd \
 && cmake . \
 && make \
 && make install \
 && cd .. \
 && rm -rf libsdd \
 && git clone https://github.com/ahamez/pnmc.git \
 && cd pnmc \
 && git submodule update --init --recursive \
 && cmake . \
 && make \
 && make install \
 && cd .. \
 && rm -rf pnmc \
>>>>>>> 39c5b73c7337f012d9241ea2b00d421b85e2649d
 && apk del .build-deps \
 && true

ENTRYPOINT pnmc
CMD [ --help ]
