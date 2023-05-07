export PROJECT_PATH="$PWD"
export COLUMBIA_PATH="${PWD}/dataset/Columbia Gaze Data Set/"
export MPII_PATH="${PWD}/dataset/MPIIFaceGaze.h5"

# train resnet10 
# ~10 hr
for i in {0..4}
do
    python train.py --model=resnet10 --augmentation --lr=1e-4 --dataset=Columbia --epoch=700 --id_test=$i
    # python train+prune.py --model=resnet10 --augmentation --lr=1e-4 --dataset=Columbia --epoch=700 --id_test=$i
done

# ~25 hr
for i in {0..14}
do
    python train.py --model=resnet10 --augmentation --lr=1e-4 --dataset=Columbia --epoch=80 --id_test=$i
done

# train resnet18

for i in {0..4}
do
    python train.py --model=resnet18 --augmentation --lr=1e-4 --dataset=Columbia --epoch=700 --id_test=$i
done

for i in {0..14}
do
    python train.py --model=resnet10 --augmentation --lr=1e-4 --dataset=Columbia --epoch=80 --id_test=$i
done