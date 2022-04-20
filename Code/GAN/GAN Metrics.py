from Code.GAN.GAN import GAN
from Data import getAttackDataIterator
if __name__ == '__main__':
    gan = GAN(5)
    gan.discriminator.load_weights('../../Checkpoints/GAN_discriminator_epoch1_avg_loss_8.545385768044166e-06.ckpt.index')
    attackIter = getAttackDataIterator(10000, 5, True, True)
    gan.test_disc(attackIter)
