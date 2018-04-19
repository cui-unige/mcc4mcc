FROM alpine

COPY Makefile  /Makefile
COPY start     /usr/bin/

RUN true \
 && apk add --no-cache --virtual .build-deps \
        autoconf \
        automake \
        bison \
        boost-dev \
        build-base \
        bzr \
        cmake \
        flex \
        git \
        sed \
 && apk add --no-cache \
        bash \
        bc \
        boost \
        parallel \
        libstdc++ \
 && git clone --depth=1 https://github.com/kerrickstaley/lp_solve.git \
    && cd lp_solve/lpsolve55 \
    && sh ccc \
    && install bin/ux64/liblpsolve55.a  /usr/lib/ \
    && install bin/ux64/liblpsolve55.so /usr/lib/ \
    && cd .. \
    && mkdir -p /usr/include/lpsolve \
    && install *.h /usr/include/lpsolve/ \
    && rm -rf lp_solve* \
 && bzr branch lp:verifypn \
    && rm verifypn/makefile* \
    && cp /Makefile verifypn/ \
    && cd verifypn \
    && rm -rf lpsolve \
    && sed -i -e "s|../lpsolve|lpsolve|g" $(find . -name "*.*pp") $(find . -name "*.h") \
    && sed -i -e "s|../lpsolve|lpsolve|g" $(find . -name "*.*pp") $(find . -name "*.h") \
    && make \
    && install -m 755 verifypn-linux64 /usr/bin/verifypn \
    && cd .. \
    && rm -rf verifypn* \
 && rm /Makefile \
 && apk del .build-deps \
 && true

COPY mcc-head  /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
