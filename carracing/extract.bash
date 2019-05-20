for i in `seq 1 13`;
do
  echo worker $i
  # on cloud:
  #xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 extract.py &
  # on macbook for debugging:
  python3 extract.py --record record_test &
  sleep 1.0
done
