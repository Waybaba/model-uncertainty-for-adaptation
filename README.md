

# install
git clone https://github.com/idiap/model-uncertainty-for-adaptation.git
git clone https://github.com/waybaba/model-uncertainty-for-adaptation.git

cp -r ../plightning/.devcontainer .devcontainer

MODEL_DIR=$UDATADIR/models/seg_models/UR
mkdir -p $MODEL_DIR
cd $MODEL_DIR
gdown 1QMpj7sPqsVwYldedZf8A5S2pT-4oENEn # Cityscape source
gdown 1KP37cQo_9NEBczm7pvq_zEmmosdhxvlF # GTA5 source
gdown 1wLffQRljXK1xoqRY64INvb2lk2ur5fEL # synthia_source.pth


ln -s $MODEL_DIR pretrained


# dataset
`
change path in utils/argparser.py for NTHU to xxx
change cityscape in datasets/new_datasets.py to root=f'{os.environ.get("UDATADIR")}/Cityscapes',
`


# train
### cityscape to crosscity
python do_segm.py --city Rio --no-src-data --freeze-classifier --unc-noise --lambda-ce 1 --lambda-ent 1  --save ./temp --lambda-ssl 0.1

### custom
python do_segm.py --city Rio --no-src-data --freeze-classifier --unc-noise --lambda-ce 1 --lambda-ent 1  --save ./temp --lambda-ssl 0.1 \
	--restore-from ./pretrained/GTA5_source.pth


--restore-from ./pretrained/GTA5_source.pth
./pretrained/Cityscapes_source_class13.pth
./pretrained/GTA5_source.pth
./pretrained/synthia_source.pth