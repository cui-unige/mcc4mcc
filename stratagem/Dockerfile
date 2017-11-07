FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        git \
        sbt \
 && apk add --no-cache \
        bash \
        openjdk8 \
 && apk del .build-deps \
 && git clone https://github.com/mundacho/stratagem.git \
 && cd stratagem \
 &&
 && cd .. \
 && rm -rf stratagem \
 && apk del .build-deps \
 && false

ENTRYPOINT its-tools
CMD [ --help ]
