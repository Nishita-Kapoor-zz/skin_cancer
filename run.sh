path="/data/nikhil/ham10000"

gpu_0() {
python main.py --path $path --version 0.0 --lr 1e-2 --gpus 0
}

gpu_1 () {
python main.py --path $path --version 0.1 --lr 1e-3 --gpus 1
}

gpu_2 () {
python main.py --path $path --version 0.2 --lr 1e-4 --gpus 2
}

gpu_3 () {
python main.py --path $path --version 0.3 --lr 1e-5 --gpus 3
}


gpu_0 &
gpu_1 &
gpu_2 &
gpu_3