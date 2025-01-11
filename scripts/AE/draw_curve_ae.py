# import matplotlib
#
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
def draw_curve_train(path, iter, train_loss, train_loss_per,train_loss_equ_affine,train_loss_equivariance_shift, train_loss_reconstruction):
    fig = plt.figure(figsize=(12, 8))  # 调整画布大小
    ax1 = fig.add_subplot(321, title="train_loss")
    ax2 = fig.add_subplot(322, title="train_loss_per")
    ax3 = fig.add_subplot(323, title="train_loss_affine")
    ax4 = fig.add_subplot(324, title="train_loss_shift")
    ax5 = fig.add_subplot(325, title='train_loss_reconstruction')

    # 绘制 train_loss 曲线
    ax1.plot(iter, [loss.cpu().detach().numpy() for loss in train_loss], 'bo-', label='train_loss: {:.3f}'.format(train_loss[-1]))
    ax1.legend()

    # 绘制 train_loss_rec 曲线
    # 绘制 train_loss_perceptual 曲线
    ax2.plot(iter, [loss.cpu().detach().numpy() for loss in train_loss_per], 'ro-',
             label='train_loss_perceptual: {:.3f}'.format(train_loss_per[-1].cpu().item()))
    ax2.legend()

    # 绘制 train_loss_equivariance_affine 曲线
    ax3.plot(iter, [loss.cpu().detach().numpy() for loss in train_loss_equ_affine], 'go-',
             label='train_loss_equivariance_affine: {:.3f}'.format(train_loss_equ_affine[-1].cpu().item()))
    ax3.legend()

    # 绘制 train_loss_equivariance_shift 曲线
    ax4.plot(iter, [loss.cpu().detach().numpy() for loss in train_loss_equivariance_shift], 'mo-',
             label='train_loss_equivariance_shift: {:.3f}'.format(train_loss_equivariance_shift[-1].cpu().item()))
    ax4.legend()

    ax5.plot(iter, [loss.cpu().detach().numpy() for loss in train_loss_reconstruction], 'co-',
             label='train_loss_reconstruction: {:.3f}'.format(train_loss_reconstruction[-1].cpu().item()))
    ax5.legend()

    # 调整布局以防止重叠
    plt.tight_layout()

    # 保存图像
    fig.savefig(path)
    plt.close(fig)



def draw_curve_test(path, step, test_fvd, test_ssim, test_psnr, test_lpips):
    fig = plt.figure(figsize=(12, 8))  # 调整画布大小
    ax1 = fig.add_subplot(221, title="Test FVD")
    ax2 = fig.add_subplot(222, title="Test SSIM")
    ax3 = fig.add_subplot(223, title="Test PSNR")
    ax4 = fig.add_subplot(224, title="Test LPIPS")

    # 绘制 test_fvd 曲线
    ax1.plot(step, test_fvd, 'bo-', label='test_fvd: {:.3f}'.format(test_fvd[-1]))
    ax1.legend()

    # 绘制 test_ssim 曲线
    ax2.plot(step, test_ssim, 'ro-', label='test_ssim: {:.3f}'.format(test_ssim[-1]))
    ax2.legend()

    # 绘制 test_psnr 曲线
    ax3.plot(step, test_psnr, 'go-', label='test_psnr: {:.3f}'.format(test_psnr[-1]))
    ax3.legend()

    # 绘制 test_lpips 曲线
    ax4.plot(step, test_lpips, 'mo-', label='test_lpips: {:.3f}'.format(test_lpips[-1]))
    ax4.legend()

    # 调整布局以防止重叠
    plt.tight_layout()

    # 保存图像
    fig.savefig(path)
    plt.close(fig)


def draw_curve(path, x_epoch, train_loss, train_prec, test_loss, test_prec, test_moda=None):
    # fig = plt.figure()
    # ax1 = fig.add_subplot(131, title="loss")
    # ax2 = fig.add_subplot(132, title="prec")
    # ax1.plot(x_epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    # ax1.plot(x_epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))
    # ax2.plot(x_epoch, train_prec, 'bo-', label='train' + ': {:.1f}'.format(train_prec[-1]))
    # ax2.plot(x_epoch, test_prec, 'ro-', label='test' + ': {:.1f}'.format(test_prec[-1]))
    #
    # ax1.legend()
    # ax2.legend()
    # if test_moda is not None:
    #     ax3 = fig.add_subplot(133, title="moda")
    #     ax3.plot(x_epoch, test_moda, 'ro-', label='test' + ': {:.1f}'.format(test_moda[-1]))
    #     ax3.legend()
    # fig.savefig(path)
    # plt.close(fig)
    fig = plt.figure()
    ax1 = fig.add_subplot(121, title="train_loss")
    ax2 = fig.add_subplot(122, title="test_loss")
    ax1.plot(x_epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    ax2.plot(x_epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))
    # ax2.plot(x_epoch, train_prec, 'bo-', label='train' + ': {:.1f}'.format(train_prec[-1]))
    # ax2.plot(x_epoch, test_prec, 'ro-', label='test' + ': {:.1f}'.format(test_prec[-1]))

    ax1.legend()
    ax2.legend()
    # if test_moda is not None:
    #     ax3 = fig.add_subplot(133, title="moda")
    #     ax3.plot(x_epoch, test_moda, 'ro-', label='test' + ': {:.1f}'.format(test_moda[-1]))
    #     ax3.legend()
    fig.savefig(path)
    plt.close(fig)

if __name__ == "__main__":
    path = '/root/home/Daijie/Semi_2D_Counting/logs/wildtrack_frame/1.jpg'
    epoch = [0, 1, 2, 3, 4, 5]
    train_loss = [1.5, 0.708, 0.845, 0.505, 0.49, 0.426]
    test_loss = [0.0, 0.0, 0., 0., 0., 0.]
    fig = plt.figure()
    ax1 = fig.add_subplot(121, title='train_loss')
    ax2 = fig.add_subplot(122, title='test_loss')
    ax1.plot(epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    ax2.plot(epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))

    fig.savefig(path)
    plt.close(fig)