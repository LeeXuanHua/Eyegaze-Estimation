export PROJECT_PATH="$PWD"

# Load dataset from Columbia Gaze
wget https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/columbia_gaze_data_set.zip
mkdir dataset
unzip -d ./dataset/ columbia_gaze_data_set.zip


# Load dataset from MPIIFaceGaze
wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip
unzip -d ./dataset/ MPIIFaceGaze_normalized.zip
mv dataset/MPIIFaceGaze_normalizad dataset/MPIIFaceGaze_normalized
python utils/preprocess_mpiifacegaze.py --dataset dataset/MPIIFaceGaze_normalized -o dataset/


export COLUMBIA_PATH="${PWD}/dataset/Columbia Gaze Data Set/"
export MPII_PATH="${PWD}/dataset/MPIIFaceGaze.h5"

mkdir results

printf "model name,learning_rate,batch_size,epoch,path to pretrain,testerror,dataset" >> results/resnet10.csv
printf "model name,learning_rate,batch_size,epoch,path to pretrain,testerror,dataset" >> results/resnet10+.csv
printf "model name,learning_rate,batch_size,epoch,path to pretrain,testerror,dataset" >> results/resnet10+P.csv

printf "model name,learning_rate,batch_size,epoch,path to pretrain,testerror,dataset" >> results/resnet18.csv
printf "model name,learning_rate,batch_size,epoch,path to pretrain,testerror,dataset" >> results/resnet18+.csv
printf "model name,learning_rate,batch_size,epoch,path to pretrain,testerror,dataset" >> results/resnet18+P.csv