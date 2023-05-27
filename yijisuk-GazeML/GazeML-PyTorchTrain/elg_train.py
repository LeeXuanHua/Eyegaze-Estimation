import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*")


def train(batch_size, eye_image_shape, epochs, version):

    import torch
    from elg.elg import ELG
    elg_model = ELG().cuda()
    # elg_model = torch.load("./models/...")

    from unityeyes import UnityEyesDataset
    train_root = os.path.join(
        "./Dataset/Train")
    train_dataset = UnityEyesDataset(
        train_root, eye_image_shape=eye_image_shape, generate_heatmaps=True, random_difficulty=True)

    val_root = os.path.join(
        "./Dataset/Val")
    val_dataset = UnityEyesDataset(
        val_root, eye_image_shape=eye_image_shape, generate_heatmaps=True, random_difficulty=True)

    start_epoch = 1
    initial_learning_rate = 1e-4

    from elg.elg_trainer import ELGTrainer
    elg_trainer = ELGTrainer(model=elg_model,
                             train_dataset=train_dataset,
                             val_dataset=val_dataset,
                             initial_learning_rate=initial_learning_rate,
                             epochs=epochs,
                             start_epoch=start_epoch,
                             batch_size=batch_size,
                             version=version)

    elg_trainer.run()


if __name__ == "__main__":
    # batch_size = eval(sys.argv[1])
    batch_size = 32
    # shape_multiplier = eval(sys.argv[2])
    shape_multiplier = 1
    # epochs = eval(sys.argv[3])
    epochs = 100

    eye_image_shape = (36*shape_multiplier, 60*shape_multiplier)

    train(batch_size, eye_image_shape, epochs,
          version=f'v0.2-{eye_image_shape}')
