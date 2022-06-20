conda create -n detctron_env
conda activate detctron_env
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install cython
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install opencv-python
cd ..
python train.py
