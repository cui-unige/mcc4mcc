FROM alpine

RUN true \
 && apk add --no-cache --virtual .build-deps \
        build-base \
        cmake \
        curl \
        git \
        python \
 && apk add --no-cache \
        bash \
        openjdk8-jre-base \
 && git clone --depth=1 https://github.com/Z3Prover/z3.git \
    && mkdir -p z3/_build \
    && cd z3/_build \
    && cmake .. \
    && make install \
    && cd ../.. \
    && rm -rf z3 \
 && curl https://yanntm.github.io/ITS-commandline/itscl_linux.tgz \
         --output itscl_linux.tgz \
    && mkdir -p /usr/share/itscl \
    && tar xf itscl_linux.tgz -C /usr/share/itscl --strip-components=1 \
    && rm -f itscl_linux.tgz \
 && apk del .build-deps \
 && true

COPY its /usr/local/bin/

VOLUME [ "/mcc-data" ]
ENTRYPOINT [ "its" ]
CMD [ "--help" ]
