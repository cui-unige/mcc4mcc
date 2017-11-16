FROM alpine

RUN true \
 && apk add --no-cache --virtual .build-deps \
        build-base \
        boost-dev \
        cmake \
        git \
 && apk add --no-cache \
        bash \
        python3 \
        libstdc++ \
 && git clone --depth=1 https://github.com/ahamez/libsdd.git \
    && mkdir -p libsdd/_build \
    && cd libsdd/_build \
    && cmake .. \
    && make install \
    && cd ../.. \
    && rm -rf libsdd \
 # && git clone --depth=1 https://github.com/ahamez/caesar.sdd.git \
 #    && mkdir -p caesar.sdd/_build \
 #    && cd caesar.sdd/_build \
 #    && cmake .. \
 #    && make install \
 #    && cd ../.. \
 #    && rm -rf caesar.sdd \
 && git clone --depth=1 https://github.com/ahamez/pnmc.git \
    && mkdir -p pnmc/_build \
    && cd pnmc/_build \
    && git submodule update --init --recursive \
    && cmake .. \
    && make install \
    && cd ../.. \
    && rm -rf pnmc \
 && apk del .build-deps \
 && true

COPY nupn.py /usr/local/bin/nupn.py
COPY pnmc.py /usr/local/bin/pnmc.py

VOLUME [ "/mcc-data" ]
ENTRYPOINT [ "pnmc.py" ]
CMD [ "--help" ]
