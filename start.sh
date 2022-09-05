#!/bin/bash
ps cax | grep gunicorn > /dev/null
if [ $? -eq 0 ]; then
  now=`date`
  echo "$now - gunicorn is running"
else
  now=`date`
  echo "$now - gunicorn is not running, starting"
  cd /home/luzian/Boolq
  source /home/luzian/miniconda3/etc/profile.d/conda.sh
  conda activate boolq
  gunicorn src.main:server -w 4 -b 127.0.0.1:8050 >& log.txt &
fi
