export LD_LIBRARY_PATH
LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/mesa:/usr/lib/x86_64-linux-gnu/mesa-egl:$LD_LIBRARY_PATH"

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 train.py --name beta5_rtd --novelty_search \
--num_worker 40 --num_worker_trial 1 --num_episode 1 --eval_steps 1 --novelty_mode a_concat \
--ns_mode NSRA --unique_id nsra