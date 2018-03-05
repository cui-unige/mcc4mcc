FROM debian:stable-slim

RUN true \
 && mkdir -p /usr/share/man/man1/ \
 && apt-get update \
 && apt-get install \
            --yes \
            --fix-missing \
            autoconf \
            automake \
            bash \
            bison \
            cmake \
            curl \
            flex \
            gcc \
            g++ \
            git \
            gperf \
            libgmpxx4ldbl \
            libgmp10 \
            libgmp-dev \
            libpopt0 \
            libpopt-dev \
            python \
            python-dev \
            z3 \
            zlib1g-dev \
            default-jre-headless \
 && curl https://yanntm.github.io/ITS-commandline/itscl_linux.tgz \
         --output itscl_linux.tgz \
    && tar xf itscl_linux.tgz \
    && mv eclipse /usr/share/itscl \
    && rm -f itscl_linux.tgz \
    && rm -f runeclipse.sh \
 && curl -L https://github.com/utwente-fmt/ltsmin/releases/download/2.1/ltsmin-2.1.tar.gz \
            --output ltsmin-2.1.tar.gz \
    && tar xf ltsmin-2.1.tar.gz \
    && cd ltsmin-2.1 \
    && sed -i".bak" "s/sysconf(_SC_LEVEL1_DCACHE_LINESIZE)/0/g" src/hre/hre_runtime.c \
    && ./configure --prefix=/usr \
    && make check \
    && make install \
    && cd .. \
    && rm -rf ltsmin-2.1* \
 && git clone https://github.com/SRI-CSL/libpoly.git \
    && mkdir -p libpoly/build \
    && cd libpoly/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
    && make \
    && make install \
    && cd ../.. \
    && rm -rf libpoly* \
 && git clone https://github.com/SRI-CSL/yices2.git \
    && cd yices2 \
    && autoconf \
    && ./configure --prefix=/usr --enable-mcsat \
    && make \
    && make install \
    && cd .. \
    && rm -rf yices2* \
 && apt-get purge \
            --yes \
            autoconf \
            automake \
            bison \
            cmake \
            curl \
            flex \
            gcc \
            g++ \
            git \
            libgmp-dev \
            libpopt-dev \
            python-dev \
            zlib1g-dev \
 && apt-get autoremove \
            --yes \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && true

COPY mcc-head /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
