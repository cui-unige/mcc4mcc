FROM debian:stable-slim

COPY from-vm/bin/checklarge/checklarge          /usr/bin/checklarge
COPY from-vm/bin/formula2lola/formula2lola      /usr/bin/formula2lola
COPY from-vm/bin/hlformula2lola/hlformula2lola  /usr/bin/hlformula2lola
COPY from-vm/bin/pnml2lola/pnml2lola            /usr/bin/pnml2lola
COPY from-vm/bin/cpn-linux                      /usr/bin/cpn-linux
COPY from-vm/bin/lola2                          /tmp/lola
COPY from-vm/bin/sara                           /tmp/sara

RUN true \
 && mkdir -p /usr/share/man/man1/ \
 && echo "APT::Install-Recommends "0";" >> /etc/apt/apt.conf.d/10norecommends \
 && echo "APT::Install-Suggests "0";"   >> /etc/apt/apt.conf.d/10norecommends \
 && apt-get update \
 && apt-get install \
        --yes \
        --fix-missing \
        bash \
        make \
        grep \
        procps \
        autoconf \
        automake \
        bison \
        flex \
        gcc \
        g++ \
        libtool \
        texinfo \
 && cd /tmp/lola \
    && ls \
    && autoreconf -fi \
    && ./configure --prefix=/usr \
    && make install \
 && cd /tmp/sara \
    && ls \
    && autoreconf -fi \
    && ./configure --prefix=/usr \
    && make install \
 && rm -rf /tmp/lola /tmp/sara \
 && apt-get purge \
            --yes \
            autoconf \
            automake \
            bison \
            flex \
            gcc \
            g++ \
            libtool \
            texinfo \
 && apt-get autoremove \
            --yes \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && true

COPY Makefile /usr/share/lola/Makefile
COPY mcc-head /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
