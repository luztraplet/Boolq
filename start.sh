source /home/luzian/miniconda3/etc/profile.d/conda.sh
conda activate boolq
gunicorn src.main:server -b 127.0.0.1:8050 >& log.txt &
