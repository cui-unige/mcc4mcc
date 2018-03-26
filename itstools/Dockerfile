FROM mccpetrinets/ltsmin

RUN true \
 && echo "APT::Install-Recommends "0";" >> /etc/apt/apt.conf.d/10norecommends \
 && echo "APT::Install-Suggests "0";"   >> /etc/apt/apt.conf.d/10norecommends \
 && apt-get update \
 && apt-get install \
            --yes \
            --fix-missing \
            autoconf \
            automake \
            bash \
            ca-certificates \
            cmake \
            curl \
            gcc \
            g++ \
            git \
            gperf \
            libgmpxx4ldbl \
            libgmp10 \
            libgmp-dev \
            make \
            python2.7 \
            python2.7-dev \
            z3 \
            default-jre-headless \
 && git clone --depth=1 https://github.com/SRI-CSL/libpoly.git \
    && mkdir -p libpoly/build \
    && cd libpoly/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
    && make \
    && make install \
    && cd ../.. \
    && rm -rf libpoly* \
 && git clone --depth=1 https://github.com/SRI-CSL/yices2.git \
    && cd yices2 \
    && autoconf \
    && ./configure --prefix=/usr --enable-mcsat \
    && make \
    && make install \
    && cd .. \
    && rm -rf yices2* \
 && curl https://yanntm.github.io/ITS-commandline/itscl_linux.tgz \
         --output itscl_linux.tgz \
    && tar xf itscl_linux.tgz \
    && mv eclipse /usr/share/itscl \
    && rm -f itscl_linux.tgz \
    && rm -f runeclipse.sh \
 && apt-get purge \
            --yes \
            autoconf \
            automake \
            ca-certificates \
            cmake \
            curl \
            gcc \
            g++ \
            git \
            gperf \
            libgmp-dev \
            make \
            python2.7-dev \
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
