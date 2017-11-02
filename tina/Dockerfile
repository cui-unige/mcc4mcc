FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        curl \
 && apk add --no-cache \
        bash \
        man \
 && curl http://projects.laas.fr/tina/binaries/tina-3.4.4-x86_64-linux.tar.gz \
         --output tina-3.4.4-x86_64-linux.tar.gz \
 && tar zxf tina-3.4.4-x86_64-linux.tar.gz \
        --directory /usr \
        --strip-components 1 \
 && rm -f tina-3.4.4-x86_64-linux.tar.gz \
 && apk del .build-deps \
 && false

ENTRYPOINT tina
CMD [ --help ]
