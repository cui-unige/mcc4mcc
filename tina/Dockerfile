FROM ubuntu

RUN true \
 && apt-get update \
 && apt-get install \
            --yes \
            bash \
            curl \
            man \
 && curl http://projects.laas.fr/tina/binaries/tina-3.4.4-x86_64-linux.tar.gz \
         --output tina-3.4.4-x86_64-linux.tar.gz \
 && tar zxf tina-3.4.4-x86_64-linux.tar.gz \
        --directory /usr \
        --strip-components 1 \
 && rm -f tina-3.4.4-x86_64-linux.tar.gz \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && true

VOLUME [ "/mcc-data" ]
ENTRYPOINT [ "tina" ]
CMD [ "--help" ]
