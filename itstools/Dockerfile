FROM debian:stable-slim

RUN true \
 && mkdir -p /usr/share/man/man1/ \
 && apt-get update \
 && apt-get install \
            --yes \
            --fix-missing \
            # autoconf \
            # automake \
            bash \
            # build-essential \
            # bison \
            # cmake \
            curl \
            # flex \
            # git \
            # python \
            # libtool \
            # libboost-dev \
            # libgmp-dev \
            libgmpxx4ldbl \
            libgmp10 \
            # libexpat1-dev \
            # libexpat1 \
            # libpopt-dev \
            # libpopt0 \
            # zlib1g-dev \
            # zlib1g \
            # libz3-dev \
            # libz3-4 \
            default-jre-headless \
 # && git clone --depth=1 https://github.com/lip6/libDDD \
 #    && cd libDDD \
 #    && autoreconf -fi \
 #    && ./configure \
 #    && make install \
 #    && cd .. \
 #    && rm -rf libDDD* \
 # && curl http://www.antlr3.org/download/antlr-3.4.tar.gz \
 #         --output antlr-3.4.tar.gz \
 #    && tar -xf antlr-3.4.tar.gz \
 #    && cp antlr-3.4/lib/antlr-3.4-complete.jar \
 #          /usr/local/lib/ \
 #    && cd ./antlr-3.4/runtime/C \
 #    && autoreconf -fi \
 #    && ./configure --enable-64bit \
 #    && make install \
 #    && cd ../../.. \
 #    && rm -rf antlr-3.4* \
 # && git clone --depth=1 https://github.com/lip6/libITS \
 #    && cd libITS \
 #    && autoreconf -fi \
 #    && export CXXFLAGS="-O3 -fno-strict-aliasing" \
 #    && ./configure --with-antlrjar=/usr/local/lib/antlr-3.4-complete.jar \
 #    && make \
 #    && make install \
 #    && cd .. \
 #    && rm -rf libITS* \
 # && curl -L https://github.com/utwente-fmt/ltsmin/releases/download/2.1/ltsmin-2.1.tar.gz \
 #        --output ltsmin-2.1.tar.gz \
 #   && tar xf ltsmin-2.1.tar.gz \
 #   && cd ltsmin-2.1 \
 #   && sed -i".bak" "s/sysconf(_SC_LEVEL1_DCACHE_LINESIZE)/0/g" src/hre/hre_runtime.c \
 #   && ./configure \
 #   && make check \
 #   && make install \
 #   && cd .. \
 #   && rm -rf ltsmin-2.1* \
 && curl https://yanntm.github.io/ITS-commandline/itscl_linux.tgz \
         --output itscl_linux.tgz \
    && tar xf itscl_linux.tgz \
    && mv eclipse /usr/share/itscl \
    && rm -f itscl_linux.tgz \
    && rm -f runeclipse.sh \
 && apt-get purge \
            --yes \
            # autoconf \
            # automake \
            # build-essential \
            # bison \
            # cmake \
            curl \
            # flex \
            # git \
            # libtool \
            # libboost-dev \
            # libgmp-dev \
            # libexpat1-dev \
            # libpopt-dev \
            # zlib1g-dev \
            # libz3-dev \
 && apt-get autoremove \
            --yes \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && true

COPY mcc-head /usr/local/bin/

VOLUME [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head" ]
CMD [ "--help" ]
