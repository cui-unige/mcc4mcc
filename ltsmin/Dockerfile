FROM alpine:edge

RUN true \
 && echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing" \
        >> /etc/apk/repositories \
 && apk add --no-cache --virtual .build-deps \
        autoconf \
        automake \
        boost-dev \
        build-base \
        cabal \
        clang \
        cmake \
        curl \
        curl-dev \
        flex \
        ghc \
        git \
        gzip \
        make \
        ncurses-dev \
        libedit-dev \
        libexecinfo-dev \
        libtbb-dev \
        openmpi-dev \
        zlib-dev \
 && apk add --no-cache \
        bash \
        ncurses \
        openmpi \
        popt \
        zlib \
        libedit \
        libexecinfo \
        libstdc++ \
        libtbb \
&& true
RUN true \
 && cabal update \
    && echo "user-install: False" >> $HOME/.cabal/config \
&& true
RUN true \
 && curl http://divine.fi.muni.cz/download/divine-4.0.18.tar.gz \
         --output divine-4.0.18.tar.gz \
    && tar xf divine-4.0.18.tar.gz \
    && rm -f divine-4.0.18.tar.gz \
    && cd divine-4.0.18 \
    && make install \
    && cd .. \
    && rm -rf divine-4.0.18 \
&& true
RUN true \
 && git clone https://github.com/jkeiren/muCRL.git \
    && cd muCRL \
    && ./configure --prefix=/usr --with-64bit CC='clang -O3' \
    && make install \
    && cd .. \
    && rm -rf muCRL \
&& true
RUN true \
 && curl http://www.mcrl2.org/download/release/mcrl2-201707.1.tar.gz \
        --output mcrl2-201707.1.tar.gz \
    && tar xf mcrl2-201707.1.tar.gz \
    && cd mcrl2-201707.1 \
    && cmake . -DMCRL2_ENABLE_GUI_TOOLS=OFF \
    && make install \
    && cd .. \
    && rm -rf mcrl2-201707.1* \
&& true
RUN true \
 && git clone https://github.com/lip6/libDDD \
    && cd libDDD \
    && autoreconf \
    && ./configure \
    && make install \
    && cd .. \
    && rm -rf libDDD \
&& true
RUN true \
 && git clone https://github.com/utwente-fmt/ltsmin.git \
    && cd ltsmin \
    && git submodule init \
    && git submodule update \
    && cd .. \
&& true
RUN true \
 && cd ltsmin/scoop \
    && cabal install hashmap \
    && ./scoopreconf \
    && ./configure \
    && make install \
    && cd ../.. \
&& true
RUN true \
 && cd ltsmin/spins \
    && true \
    && cd ../.. \
&& true
RUN true \
 && cd ltsmin \
    && autoreconf \
    && ./configure --prefix=/usr --disable-dependency-tracking \
    && make install \
    && cd .. \
    && rm -rf ltsmin \
&& true
RUN true \
 && apk del .build-deps \
 && false

ENTRYPOINT ltsmin
CMD [ --help ]
