#mkdir LineMOD_Dataset
#cd LineMOD_Dataset


for file in `ls`; do unzip $file; done;
cd ..
wget - c 'http://images.cocodataset.org/zips/val2017.zip'
unzip val2017.zip
