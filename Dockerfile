FROM ubuntu:20.04

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY ws /ws

WORKDIR /ws

ENTRYPOINT ["/entrypoint.sh"]