FROM alpine

RUN true \
 && apk add --no-cache \
        bash \
        coreutils \
        openjdk8-jre-base \
 && true

COPY greatspn       /usr/local/share/gspn
COPY MccCtlConv.jar /usr/local/share/gspn/bin/
COPY mcc-head       /usr/local/bin/

VOLUME [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head" ]
CMD [ "--help" ]
