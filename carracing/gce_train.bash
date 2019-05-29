export LD_LIBRARY_PATH
LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/mesa:/usr/lib/x86_64-linux-gnu/mesa-egl:$LD_LIBRARY_PATH"

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 train.py --name baseline \
--num_worker 12 --num_worker_trial 3 --num_episode 16 --eval_steps 5 --unique_id baseline