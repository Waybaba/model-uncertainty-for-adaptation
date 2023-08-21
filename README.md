

# install
git clone https://github.com/idiap/model-uncertainty-for-adaptation.git
git clone https://github.com/waybaba/model-uncertainty-for-adaptation.git

cp -r ../plightning/.devcontainer .devcontainer

MODEL_DIR=$UDATADIR/models/seg_models/UR
mkdir -p $MODEL_DIR
cd $MODEL_DIR
gdown 1QMpj7sPqsVwYldedZf8A5S2pT-4oENEn
ln -s $MODEL_DIR pretrained


# train
python do_segm.py --city Rio --no-src-data --freeze-classifier --unc-noise --lambda-ce 1 --lambda-ent 1  --save ./temp --lambda-ssl 0.1