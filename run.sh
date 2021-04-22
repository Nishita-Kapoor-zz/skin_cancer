path="/data/nikhil/ham10000"

gpu_0() {
python main.py --path $path --version 0.0 --lr 1e-2 --gpus 0
python main.py --path $path --version 1.0 --gpus 0
python main.py --path $path --version 2.1 --gpus 0 --loss focal

}

gpu_1 () {
python main.py --path $path --version 0.1 --lr 1e-3 --gpus 1
python main.py --path $path --version 1.1 --gpus 1 --loss focal
python main.py --path $path --version 2.2 --gpus 1 --loss weighted_ce

}

gpu_2 () {
python main.py --path $path --version 0.2 --lr 1e-4 --gpus 2
python main.py --path $path --version 1.2 --gpus 2 --loss weighted_ce
}

gpu_3 () {
python main.py --path $path --version 0.3 --lr 1e-5 --gpus 3
}


# gpu_0 &
# gpu_1 &
# gpu_2 &
# gpu_3