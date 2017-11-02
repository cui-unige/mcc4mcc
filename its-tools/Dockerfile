FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        git \
 && apk add --no-cache \
        bash \
        openjdk8 \
 && apk del .build-deps \
 && false

ENTRYPOINT its-tools
CMD [ --help ]
