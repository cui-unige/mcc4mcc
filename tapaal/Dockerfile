FROM alpine:edge

RUN true \
 && apk add --no-cache --virtual .build-deps \
        curl \
        unzip \
 && apk add --no-cache \
        bash \
        openjdk8 \
 && curl http://www.tapaal.net/fileadmin/download/tapaal-3.3/tapaal-3.3.0-linux64.zip \
         --output tapaal-3.3.0-linux64.zip \
 && unzip tapaal-3.3.0-linux64.zip \
 && mv tapaal-3.3.0-linux64.zip /opt/tapaal \
 && ln -s /opt/tapaal/tapaal /usr/bin/tapaal \
 && rm -f tapaal-3.3.0-linux64.zip \
 && apk del .build-deps \
 && false

ENTRYPOINT tapaal
CMD [ --help ]
