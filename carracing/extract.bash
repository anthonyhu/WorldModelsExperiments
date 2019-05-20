export LD_LIBRARY_PATH
LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/mesa:/usr/lib/x86_64-linux-gnu/mesa-egl:$LD_LIBRARY_PATH"

for i in `seq 1 2`;
do
  echo worker $i
  # on cloud:
  xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 extract.py --record record_test &
  # on macbook for debugging:
  #python3 extract.py --record record_test &
  sleep 1.0
done
