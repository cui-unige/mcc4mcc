FROM debian:stable-slim

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
            bison \
            cmake \
            curl \
            flex \
            gcc \
            g++ \
            git \
            gperf \
            latexmk \
            libtool \
            libltdl-dev \
            libgmpxx4ldbl \
            libgmp10 \
            libgmp-dev \
            libpopt0 \
            libpopt-dev \
            libxml2 \
            libxml2-dev \
            make \
            perl \
            pdf2svg \
            pkgconf \
            python \
            python-dev \
            python3 \
            python3-dev \
            texlive-latex-extra \
            texlive-fonts-extra \
            texlive-fonts-recommended \
            texlive-science \
            unzip \
            zlib1g-dev \
 && git clone --depth=1 https://github.com/trolando/sylvan.git \
    && mkdir -p sylvan/build \
    && cd sylvan/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
    && make install \
    && cd ../.. \
    && rm -rf sylvan \
 && git clone --depth=1 https://github.com/lip6/libDDD.git \
    && cd libDDD \
    && autoreconf -vfi \
    && ./configure --prefix=/usr \
    && make install \
    && cd .. \
    && rm -rf libDDD \
 && git clone --depth=1 https://gitlab.lrde.epita.fr/spot/spot.git \
    && cd spot \
    && autoreconf -vfi \
    && ./configure --prefix=/usr \
                   --disable-python \
    && make install \
    && cd .. \
    && rm -rf spot \
 && git clone --depth=1 https://github.com/utwente-fmt/ltsmin.git \
    && cd ltsmin \
    && git submodule update --init \
    && ./ltsminreconf \
    # && sed -i".bak" "s/sysconf(_SC_LEVEL1_DCACHE_LINESIZE)/0/g" src/hre/hre_runtime.c \
    && autoreconf -vfi \
    && ./configure --prefix=/usr \
    && make install \
    && cd .. \
    && rm -rf ltsmin* \
 && apt-get purge \
            --yes \
            autoconf \
            automake \
            bison \
            ca-certificates \
            cmake \
            curl \
            flex \
            gcc \
            g++ \
            git \
            gperf \
            latexmk \
            libtool \
            libltdl-dev \
            libgmp-dev \
            libpopt-dev \
            libxml2-dev \
            make \
            pdf2svg \
            pkgconf \
            python-dev \
            python3-dev \
            texlive-latex-extra \
            texlive-fonts-extra \
            texlive-fonts-recommended \
            texlive-science \
            unzip \
            zlib1g-dev \
 && apt-get autoremove \
            --yes \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && true

COPY formulizer.py /usr/bin/formulizer
COPY mcc-head      /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
