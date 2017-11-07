FROM mcc/ltsmin

RUN true \
 && apk add --no-cache --virtual .build-deps \
        curl \
        git \
        tree \
 && apk add --no-cache \
        bash \
        openjdk8-jre-base \
 && curl https://yanntm.github.io/ITS-commandline/itscl_linux.tgz \
         --output itscl_linux.tgz \
 && tar xf itscl_linux.tgz \
 && rm -f itscl_linux.tgz \
 && curl https://yanntm.github.io/ITS-Tools-Dependencies/z3.tgz \
         --output z3.tgz \
 && tar xf z3.tgz \
 && rm -f z3.tgz \
 && curl https://yanntm.github.io/LTSmin-BinaryBuilds/ltsmin_linux_64.tar.gz \
         --output ltsmin_linux_64.tar.gz \
 && tar xf ltsmin_linux_64.tar.gz \
 && rm -f ltsmin_linux_64.tar.gz \
 && apk del .build-deps \
 && false

ENTRYPOINT its-tools
CMD [ --help ]
