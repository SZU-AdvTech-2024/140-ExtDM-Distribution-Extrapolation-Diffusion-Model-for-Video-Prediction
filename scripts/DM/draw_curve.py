# import matplotlib
#
# matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch


def process_losses(loss_list):
    if isinstance(loss_list, list):
        return [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in loss_list]
    raise TypeError(f"Expected a list of losses, but got {type(loss_list)}")
def draw_curve_train(path, iter, train_loss, train_loss_rec,train_loss_wrap,train_loss_total):


    fig = plt.figure(figsize=(12, 8))  # 调整画布大小
    ax1 = fig.add_subplot(231, title="train_loss")
    ax2 = fig.add_subplot(232, title="train_loss_rec")
    ax3 = fig.add_subplot(233, title="train_loss_wrap")
    ax5 = fig.add_subplot(234, title="train_loss_total")
    # ax4 = fig.add_subplot(234, title='train_loss_vgg')
    # ax5 = fig.add_subplot(235, title='trian_loss_pyra')
    if isinstance(iter, torch.Tensor):
        iter = iter.cpu().numpy()
    elif not isinstance(iter, (list, tuple)):
        raise TypeError(f"'iter' must be a Tensor, list, or tuple, got {type(iter)}.")
    train_loss = process_losses(train_loss)
    train_loss_rec = process_losses(train_loss_rec)
    train_loss_wrap = process_losses(train_loss_wrap)
    # train_loss_vgg = process_losses(train_loss_vgg)
    train_loss_total = process_losses(train_loss_total)

    ax1.plot(iter, train_loss, 'bo-',
             label=f'train_loss: {train_loss[-1]:.3f}')
    ax1.legend()

    # 绘制 train_loss_rec 曲线
    ax2.plot(iter, train_loss_rec, 'ro-',
             label=f'train_loss_rec: {train_loss_rec[-1]:.3f}')
    ax2.legend()

    # 绘制 train_loss_wrap 曲线
    ax3.plot(iter, train_loss_wrap, 'go-',
             label=f'train_loss_wrap: {train_loss_wrap[-1]:.3f}')
    ax3.legend()

    # 绘制 train_loss_vgg 曲线
    # ax4.plot(iter, train_loss_vgg, 'ko-',
    #          label=f'train_loss_vgg: {train_loss_vgg[-1]:.3f}')
    # ax4.legend()

    # 绘制 train_loss_total 曲线
    ax5.plot(iter, train_loss_total, 'ko-',
             label=f'train_loss_total: {train_loss_total[-1]:.3f}')
    ax5.legend()

    # 自动调整子图布局
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

    # # 确保 iter 是 NumPy 数组
    # iter = iter.cpu().numpy() if isinstance(iter, torch.Tensor) else iter
    #
    # # 绘制 train_loss 曲线
    # ax1.plot(iter, [loss.cpu().item() for loss in train_loss], 'bo-',
    #          label='train_loss: {:.3f}'.format(train_loss[-1].cpu().item()))
    # ax1.legend()
    #
    # # 绘制 train_loss_rec 曲线
    # ax2.plot(iter, [loss.cpu().item() for loss in train_loss_rec], 'ro-',
    #          label='train_loss_rec: {:.3f}'.format(train_loss_rec[-1].cpu().item()))
    # ax2.legend()
    #
    # # 绘制 train_loss_wrap 曲线
    # ax3.plot(iter, [loss.cpu().item() for loss in train_loss_wrap], 'go-',
    #          label='train_loss_wrap: {:.3f}'.format(train_loss_wrap[-1].cpu().item()))
    # ax3.legend()
    #
    # ax4.plot(iter, [loss.cpu().item() for loss in train_loss_vgg], 'ko-',
    #          label='train_loss_vgg: {:.3f}'.format(train_loss_vgg[-1].cpu().item()))
    # ax4.legend()
    #
    # # 绘制 train_loss_total 曲线
    # ax5.plot(iter, [loss.cpu().item() for loss in train_loss_total], 'ko-',
    #          label='train_loss_total: {:.3f}'.format(train_loss_total[-1].cpu().item()))
    # ax5.legend()
    #
    # plt.tight_layout()  # 自动调整子图布局
    # plt.savefig(path)
    # plt.close(fig)




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