FROM mccpetrinets/itstools

COPY from-vm/its-tools /usr/share/its-tools

RUN true \
 && true

COPY mcc-head /usr/bin

WORKDIR    /mcc-data
VOLUME     [ "/mcc-data" ]
ENTRYPOINT [ "mcc-head"  ]
CMD        [ "--help"    ]
