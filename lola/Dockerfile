FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        autoconf \
        automake \
        bison \
        curl \
        flex \
        gengetopt \
        libtool \
        make \
        subversion \
        texinfo \
 && apk add --no-cache \
        bash \
        build-base \
        grep \
        libstdc++ \
        man \
 && curl https://www2.informatik.hu-berlin.de/sam/kimwitu++/kimwitu++-2.3.11.tar.gz \
      --output kimwitu++-2.3.11.tar.gz \
    && tar zxf kimwitu++-2.3.11.tar.gz \
    && cd kimwitu++-2.3.11 \
    && ./configure --prefix=/usr \
    && make install \
    && cd .. \
    && rm -rf kimwitu++-2.3.11* \
    && rm -rf /usr/share/man/man1 \
    && touch /usr/include/fpu_control.h \
 && curl http://service-technology.org/files/lola/lola-2.0.tar.gz \
      --output lola-2.0.tar.gz \
    && tar zxf lola-2.0.tar.gz \
    && cd lola-2.0 \
    && find src/ -type f -exec sed -i".bak" "s/u_short/unsigned short/g" {} \; \
    && sed -i".bak" "s/defined(__linux__)/defined(_FPU_GETCW)/g" src/Siphon/lola2minisat.cc \
    && sed -i".bak" "s/defined(__linux__)/defined(_FPU_GETCW)/g" src/ThirdParty/minisat/utils/System.h \
    && autoreconf -fi \
    && ./configure --prefix=/usr \
    && make install \
    && cd .. \
    && rm -rf lola-2.0* \
 && cd /tmp/lola/sara \
    && autoreconf -fi \
    && ./configure --prefix=/usr \
    && make install \
    && cd \
 && cd /tmp/lola/checklarge \
    && rm -f checklarge *.tab.c lex.yy.c \
    && make \
    && install checklarge /usr/bin/ \
    && cd \
 && cd /tmp/lola/formula2lola \
    && rm -f formula2lola *.tab.c lex.yy.c \
    && sed -i '/stdout/d' formula2lola.y \
    && make \
    && install formula2lola /usr/bin/ \
    && cd \
 # && cd /tmp/lola/hlformula2lola \
 #    && rm -f hlformula2lola *.tab.c lex.yy.c \
 #    && make \
 #    && install hlformula2lola /usr/bin/ \
 #    && cd \
 && cd /tmp/lola/pnml2lola \
    && rm -f pnml2lola *.tab.c lex.yy.c \
    && make \
    && install pnml2lola /usr/bin/ \
    && cd \
 && rm -rf /tmp/lola \
 && apk del .build-deps \
 && true

COPY from-vm/bin/checklarge/checklarge          /usr/bin/checklarge
COPY from-vm/bin/formula2lola/formula2lola      /usr/bin/formula2lola
COPY from-vm/bin/hlformula2lola/hlformula2lola  /usr/bin/hlformula2lola
COPY from-vm/bin/pnml2lola/pnml2lola            /usr/bin/pnml2lola
COPY from-vm/bin/sara/sara                      /usr/bin/sara
COPY from-vm/bin/cpn-linux                      /usr/bin/cpn-linux
COPY Makefile /usr/share/lola/Makefile
COPY mcc-head /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
