git clone https://github.com/gulang2019/SLOServe.git
cd SLOServe
git submodule update --init

conda create --name myenv python=3.10  -y
conda activate myenv
pip install -r requirements.txt

python Dataset/download_dataset.py

cd 3rdparty/vllm
git pull origin main
VLLM_USE_PRECOMPILED=1 python3 -m pip install --editable .
cd -

cd csrc
python3 -m pip install -e . --no-build-isolation
cd -