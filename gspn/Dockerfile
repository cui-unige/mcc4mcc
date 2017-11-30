FROM debian:stable-slim

COPY usr-local /tmp/usr-local

RUN true \
 && mv /tmp/usr-local/include/* /usr/local/include/ \
 && mv /tmp/usr-local/lib/*     /usr/local/lib/ \
 && mv /tmp/usr-local/GreatSPN  /usr/local/ \
 && rm -rf /tmp/usr-local \
 && mkdir -p /usr/share/man/man1/ \
 && apt-get update \
 && apt-get install \
            --yes \
            --fix-missing \
            # autoconf \
            # automake \
            bash \
            # build-essential \
            default-jre-headless \
            # git \
            # libtool \
            # libgmp-dev \
            libgmpxx4ldbl \
            libgmp10 \
 # && git clone git://git.code.sf.net/p/meddly/code-git meddly \
 #    && cd meddly \
 #    && autoreconf -fi \
 #    && ./configure --prefix=/usr/local \
 #    && make install \
 #    && cd .. \
 #    && rm -rf meddly* \
 && sed -i \
        -e 's/assistive_technologies/#assistive_technologies/g' \
        /etc/java-8-openjdk/accessibility.properties \
 # && apt-get purge \
 #            --yes \
 #            autoconf \
 #            automake \
 #            build-essential \
 #            git \
 #            libtool \
 #            libgmp-dev \
 && apt-get autoremove \
            --yes \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && true

# COPY greatspn       /usr/local/share/gspn
COPY MccCtlConv.jar /usr/local/GreatSPN/bin/
COPY mcc-head       /usr/local/bin/

VOLUME [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head" ]
CMD [ "--help" ]
