FROM alpine

COPY mcc.lua \
     /mcc-install/
COPY mcc-dev-0.rockspec \
     /mcc-install/

RUN true \
 && echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing" \
         >> /etc/apk/repositories \
 && apk add --no-cache --virtual .build-deps \
        build-base \
        curl \
        git \
        lua5.3-dev \
        luarocks5.3 \
        openssl-dev \
 && apk add --no-cache \
        bash \
        dockerize \
        lua5.3 \
 && cd /mcc-install \
 && luarocks-5.3 make mcc-dev-0.rockspec \
 && rm -rf /mcc-install \
 && apk del .build-deps \
 && true

VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc"       ]
CMD        [ "--help"    ]
WORKDIR    /mcc-data
