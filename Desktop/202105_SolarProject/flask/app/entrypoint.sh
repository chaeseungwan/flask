#!/bin/bash


# run cron
chmod 644 /etc/cron.d/root
chown root:root /etc/cron.d/root

cron

uwsgi uwsgi.ini
