FROM alpine

COPY mcc.lua \
     /mcc-install/
COPY mcc-dev-0.rockspec \
     /mcc-install/

RUN apk add --no-cache --virtual .build-deps \
        build-base \
        curl \
        git \
        lua5.3-dev \
        luarocks5.3 \
        openssl-dev \
 && apk add --no-cache --repository http://dl-3.alpinelinux.org/alpine/edge/testing/ \
        dockerize \
 && apk add --no-cache \
        bash \
        docker \
        lua5.3 \
        openssl \
 && luarocks-5.3 install luasec \
 && cd /mcc-install \
 && luarocks-5.3 make mcc-dev-0.rockspec \
 && rm -rf /mcc-install \
 && apk del .build-deps \
 && true

ENTRYPOINT ["mcc"]
CMD ["--help"]
