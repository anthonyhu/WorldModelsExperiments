export LD_LIBRARY_PATH
LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/mesa:/usr/lib/x86_64-linux-gnu/mesa-egl:$LD_LIBRARY_PATH"

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 train.py --name beta5_rtd --num_worker 40 --unique_id verif