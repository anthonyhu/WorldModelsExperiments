for i in `seq 1 12`;
do
  echo worker $i
  # on cloud:
  #export LD_LIBRARY_PATH
  #LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/mesa:/usr/lib/x86_64-linux-gnu/mesa-egl:$LD_LIBRARY_PATH"
  #xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 extract.py --record record_test &
  # on macbook for debugging:
  python3 extract.py --record record_from_trained_baseline --full_model_path /data/cvfs/ah2029/datasets/gym/carracing/saved_models/baseline_vae1.2k_rnn/ &
  sleep 1.0
done
