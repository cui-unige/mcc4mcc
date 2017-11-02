FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        build-base \
        clang \
        curl \
        make \
        texinfo \
 && apk add --no-cache \
        bash \
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
 && ./configure CC=clang CXX=clang++ --prefix=/usr \
 && make install \
 && cd .. \
 && rm -rf lola-2.0* \
 && apk del .build-deps \
 && true

ENTRYPOINT lola
CMD [ --help ]
