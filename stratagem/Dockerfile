FROM alpine

RUN true \
 && echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing" \
         >> /etc/apk/repositories \
 && apk add --no-cache --virtual .build-deps \
        git \
        sbt \
 && apk add --no-cache \
        bash \
        openjdk8 \
 && apk del .build-deps \
 && git clone --depth=1 https://github.com/mundacho/stratagem.git \
 && cd stratagem \
 && false \
 && cd .. \
 && rm -rf stratagem \
 && apk del .build-deps \
 && false

VOLUME [ "/mcc-data" ]
ENTRYPOINT [ "stratagem" ]
CMD [ "--help" ]
