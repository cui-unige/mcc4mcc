FROM alpine

RUN true \
 && echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing" \
         >> /etc/apk/repositories \
 && apk add --no-cache --virtual .build-deps \
        curl \
        git \
        openjdk8 \
 && apk add --no-cache \
        bash \
        openjdk8-jre-base \
 && curl -L https://github.com/sbt/sbt/releases/download/v0.13.17/sbt-0.13.17.tgz \
      --output sbt-0.13.17.tgz \
    && tar xf sbt-0.13.17.tgz \
         --directory /usr/local/ \
         --strip-components 1 \
    && rm -f sbt-0.13.17.tgz \
 && git clone --depth=1 https://github.com/mundacho/stratagem.git \
    && cd stratagem/stratagem \
    && sbt package -no-share \
    && mkdir -p /usr/local/share/stratagem/ \
    && for i in $(find . -name "*.jar"); \
       do mv $i /usr/local/share/stratagem/$(basename $i); \
       done \
    && cd ../.. \
    && rm -rf stratagem \
 && apk del .build-deps \
 && true

COPY mcc-head /usr/bin/

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
