# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com
from __future__ import absolute_import, division, print_function
from monodepth_model_original import *
from monodepth_dataloader_train_noise import *
from average_gradients import *
import tensorflow.contrib.slim as slim
import argparse
import time
import tensorflow as tf
from PIL import Image
from skimage import measure
# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='training or test', default='test')
parser.add_argument('--model_name',                type=str,   help='model name', default='/media/weixingming/Samsung_T5/LF/ckpt/model-110000')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='resnet50')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, HCI, or Stanford', default='HCI')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='/media/weixingming/Samsung_T5/LF/test_set')#/home/sjw/Desktop/
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames txt file', default='./utils/filenames/flowers_train_set_49_allin.txt')
parser.add_argument('--test_file',                 type=str,   help='path to the test file',           default='./utils/filenames/new_TCW_test_set_49.txt')
parser.add_argument('--input_height',              type=int,   help='input height', default=352)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=10)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=2)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=2)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default = 4)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='/home/sjw/Desktop/extra_hard_disk/tensorflow_temp/monodepth_multi/log/output')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='/home/sjw/Desktop/extra_hard_disk/tensorflow_temp/monodepth_multi/log')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='/media/weixingming/Samsung_T5/LF/ckpt/model-110000')#/home/sjw/tensorflow_temp/monodepth_multi/log/monodepth/multi_flower_add_inputnet/model-105000
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', default=False)
parser.add_argument('--num_output',                type=int,   help='the number of output', default=1)
args = parser.parse_args()


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def train(params):
    """Training loop."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)
        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        # 学习率更新策略
        learning_rate = tf.train.polynomial_decay(learning_rate=args.learning_rate,
                                                  global_step=global_step,
                                                  decay_steps=30000,
                                                  end_learning_rate=args.learning_rate/10,
                                                  power=2,
                                                  cycle=True)

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))
        # 初始化dataset
        dataloader = MonodepthDataloader(data_path=args.data_path,
                                         filenames_file=args.filenames_file,
                                         params=params,
                                         mode='train',
                                         num_output=args.num_output)
        dataset = dataloader.dataset
        input_images, label_image, target_pos, onehot_pos, input_imgs_orig = dataset.make_one_shot_iterator().get_next()

        input_images.set_shape([params.batch_size, params.height, params.width, 27])
        label_image.set_shape([params.batch_size, params.height, params.width, 3])
        target_pos.set_shape([params.batch_size, 2])
        onehot_pos.set_shape([params.batch_size, params.height, 49, 1])
        input_imgs_orig.set_shape([params.batch_size, params.height, params.width, 27])

        tower_grads = []
        tower_losses = []
        reuse_variables = tf.AUTO_REUSE
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                input_splits = tf.split(input_images, args.num_gpus, 0)[i]
                label_splits = tf.split(label_image, args.num_gpus, 0)[i]
                target_pos_splits = tf.split(target_pos, args.num_gpus, 0)[i]
                onehot_pos_splits = tf.split(onehot_pos, args.num_gpus, 0)[i]
                input_orig_splits = tf.split(input_imgs_orig, args.num_gpus, 0)[i]
                with tf.device('/gpu:%d' % i):
                    model = MonodepthModel(params=params,
                                           mode='train',
                                           input_image=input_splits,
                                           input_orig_image=input_orig_splits,
                                           label_image=label_splits,
                                           target_pos=target_pos_splits,
                                           target_onehot_pos=onehot_pos_splits,
                                           reuse_variables=reuse_variables,
                                           model_index=i,
                                           num_output=args.num_output)
                    # trainable_vars = tf.trainable_variables()
                    # freeze_conv_var_list = [t for t in trainable_vars if t.name.startswith(u'model/color_net')]
                    # total_num_parameters = 0
                    # for variable in freeze_conv_var_list:
                    #     total_num_parameters += np.array(variable.get_shape().as_list()).prod()
                    # print("number of trainable parameters: {}".format(total_num_parameters))
                    loss = model.total_loss
                    tower_losses.append(loss)
                    reuse_variables = True
                    # grads = opt_step.compute_gradients(loss, var_list=freeze_conv_var_list)
                    grads = opt_step.compute_gradients(loss)
                    tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)
        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver(max_to_keep=30)

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        # #获取需要加载的变量
        # varialbes = tf.contrib.framework.get_variables_to_restore()
        # #删除新增加的input_net的变量
        # variables_to_restore = [v for v in varialbes if v.name.split('/')[0] != 'input_net']
        # saver_ori = tf.train.Saver(variables_to_restore)
        # variables_to_restore = [v for v in varialbes if v.name.split('/')[0] == 'input_net']
        # saver_input_net = tf.train.Saver(variables_to_restore)
        # LOAD CHECKPOINT IF SET
        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        if args.checkpoint_path != '':
            # saver_ori.restore(sess, args.checkpoint_path.split(".")[0])
            # saver_input_net.restore(sess,'/home/sjw/tensorflow_temp/monodepth_multi/log/monodepth/multi_flower_add_inputnet/model-1000')
            # all_var = tf.trainable_variables()
            # resnet_encode_var = [t for t in all_var if t.name.startswith(u'model/encoder') and 'model/encoder/Conv/' not in t.name]
            # resnet_decode_var = [t for t in all_var if t.name.startswith(u'model/decoder') and 'model/decoder/Conv_15' not in t.name]
            # restore_saver = tf.train.Saver(var_list=resnet_encode_var)
            # restore_saver.restore(sess, args.checkpoint_path.split(".")[0])
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])
            print('load from {}'.format(args.checkpoint_path))
            print('load model successful!')
            if args.retrain:
                sess.run(global_step.assign(0))
            else:
                sess.run(global_step.assign(int(args.checkpoint_path.split("-")[-1]) + 1))
        else:
            # all_var = tf.trainable_variables()
            # disp_net_var = [t for t in all_var if t.name.startswith(u'model/encoder') or t.name.startswith(u'model/decoder')]
            # saver_disp_net = tf.train.Saver(var_list=disp_net_var)
            # saver_disp_net.restore(sess,
            #                         '/home/sjw/tensorflow_temp/monodepth_multi/log/monodepth/multi_to_one_multitask_ssim/model-60000')
            # print('restore dispnet')

            print('don\'t load model')

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()

        for step in range(start_step, num_total_steps+1):
            before_op_time = time.time()
            _, loss_value = sess.run([apply_gradient_op, total_loss])
            duration = time.time() - before_op_time

            if step and step % 100 == 0:#100
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and (step % 20000 == 0 or step == 110000):
                print('save model')
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)
                # evaluation(sess, step)
        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)


def evaluation(sess, times):
        """eval function."""
        params = monodepth_parameters(
            encoder=args.encoder,
            height=args.input_height,
            width=args.input_width,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            num_epochs=args.num_epochs,
            do_stereo=args.do_stereo,
            wrap_mode=args.wrap_mode,
            use_deconv=args.use_deconv,
            alpha_image_loss=args.alpha_image_loss,
            disp_gradient_loss_weight=args.disp_gradient_loss_weight,
            lr_loss_weight=args.lr_loss_weight,
            full_summary=args.full_summary,
            num_gpus=args.num_gpus)

        dataloader = MonodepthDataloader(data_path=args.data_path,
                                         filenames_file=args.test_file,
                                         params=params,
                                         mode='test',
                                         num_output=args.num_output)
        dataset = dataloader.dataset
        input_images, label_image, target_pos = dataset.make_one_shot_iterator().get_next()

        input_images.set_shape([1, params.height, params.width, 27])
        label_image.set_shape([1, params.height, params.width, 3])
        target_pos.set_shape([1, 2])


        reuse_variables = tf.AUTO_REUSE
        model = MonodepthModel(params=params,
                               mode='test',
                               input_image=input_images,
                               label_image=label_image,
                               target_pos=target_pos,
                               reuse_variables=reuse_variables,
                               num_output=args.num_output)

        num_test_samples = count_text_lines(args.test_file)//49
        output_directory = args.output_directory
        print('Now evalue {} samples'.format(num_test_samples))

        for num in range(num_test_samples):
            print('Sample No.' + str(num + 1))

            path_to_disp_est = output_directory + '/' + args.model_name + '/model_%d' % times + '/disp_est' + '/sample_%04d' % num
            path_to_img_est = output_directory + '/' + args.model_name + '/model_%d' % times + '/est_img' + '/sample_%04d' % num
            path_to_gt = output_directory + '/' + args.model_name + '/model_%d' % times + '/gt_img' + '/sample_%04d' % num

            isExists1 = os.path.exists(path_to_disp_est)
            isExists2 = os.path.exists(path_to_img_est)
            isExists3 = os.path.exists(path_to_gt)

            if not isExists1:
                os.makedirs(path_to_disp_est)
            if not isExists2:
                os.makedirs(path_to_img_est)
            if not isExists3:
                os.makedirs(path_to_gt)

            SSIM_val = []
            PSNR_val = []
            MSE_val = []
            NRMSE_val = []
            for i in range(49):
                path_store_img_est = path_to_img_est + '/%03d.png' % i
                path_store_disp_est = path_to_disp_est + '/%03d.png' % i
                path_store_gt = path_to_gt + '/%03d.png' % i

                disp_est, img_est, gt = sess.run([model.disp, model.color_output, model.label_image])

                imgest = Image.fromarray(np.clip(img_est[0] * 255, 0, 255).astype(np.uint8))
                imggt = Image.fromarray(np.clip(gt[0] * 255, 0, 255).astype(np.uint8))
                imgdispest = Image.fromarray(np.clip((disp_est[0, :, :, 0] + 4) / 8 * 255, 0, 255).astype(np.uint8),
                                             mode='L')

                imgest.save(path_store_img_est)
                imggt.save(path_store_gt)
                imgdispest.save(path_store_disp_est)

                if i not in [0, 3, 6, 21, 24, 27, 42, 45, 48]:
                    SSIM_val.append(measure.compare_ssim(np.clip(gt[0] * 255, 0, 255), np.clip(img_est[0] * 255, 0, 255),
                                             data_range=255, multichannel=True))
                    PSNR_val.append(measure.compare_psnr(np.clip(gt[0] * 255, 0, 255), np.clip(img_est[0] * 255, 0, 255),
                                         data_range=255))
                    MSE_val.append(measure.compare_mse(np.clip(gt[0] * 255, 0, 255), np.clip(img_est[0] * 255, 0, 255)))
                    NRMSE_val.append(measure.compare_nrmse(np.clip(gt[0] * 255, 0, 255), np.clip(img_est[0] * 255, 0, 255)))
            ssim_mean = np.mean(SSIM_val)
            psnr_mean = np.mean(PSNR_val)
            mse_mean = np.mean(MSE_val)
            nrmse_mean = np.mean(NRMSE_val)
            with open(output_directory + '/' + args.model_name + '/psnr_ssim.txt', 'a') as fileobject:
                if num == 0:
                    title = '{:<20}{:<30}{:<30}{:<30}{:<30}\n'.format('step' + str(times), 'SSIM', 'PSNR', 'MSE', 'NRMSE')
                    fileobject.write(title)
                lines = '{:<20}{:<30}{:<30}{:<30}{:<30}\n'.format('Sample' + str(num) + ':', ssim_mean, psnr_mean,
                                                                mse_mean, nrmse_mean)
                fileobject.write(lines)


def test(params):
    """Test function."""
    times = 70000

    dataloader = MonodepthDataloader(data_path=args.data_path,
                                     filenames_file=args.test_file,
                                     params=params,
                                     mode='test',
                                     num_output=args.num_output)
    dataset = dataloader.dataset
    input_images, label_image, target_pos = dataset.make_one_shot_iterator().get_next()

    input_images.set_shape([1, params.height, params.width, 27])
    label_image.set_shape([1, params.height, params.width, 3])
    target_pos.set_shape([1, 2])

    reuse_variables = tf.AUTO_REUSE
    model = MonodepthModel(params=params,
                           mode='test',
                           input_image=input_images,
                           label_image=label_image,
                           target_pos=target_pos,
                           reuse_variables=reuse_variables,
                           num_output=args.num_output)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver(max_to_keep=30)

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
        print("restore from {}".format(restore_path))
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.test_file) // 49
    print('Now evalue {} samples'.format(num_test_samples))

    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory + '/test'

    for num in range(num_test_samples):
        print('Sample No.' + str(num + 1))

        path_to_disp_est = output_directory + '/' + args.model_name + '/model_%d' % times + '/disp_est' + '/sample_%04d' % num
        path_to_img_est = output_directory + '/' + args.model_name + '/model_%d' % times + '/est_img' + '/sample_%04d' % num
        path_to_gt = output_directory + '/' + args.model_name + '/model_%d' % times + '/gt_img' + '/sample_%04d' % num

        isExists1 = os.path.exists(path_to_disp_est)
        isExists2 = os.path.exists(path_to_img_est)
        isExists3 = os.path.exists(path_to_gt)

        if not isExists1:
            os.makedirs(path_to_disp_est)
        if not isExists2:
            os.makedirs(path_to_img_est)
        if not isExists3:
            os.makedirs(path_to_gt)

        SSIM_val = []
        PSNR_val = []
        MSE_val = []
        NRMSE_val = []
        for i in range(49):
            path_store_img_est = path_to_img_est + '/%03d.png' % i
            path_store_disp_est = path_to_disp_est + '/%03d.png' % i
            path_store_gt = path_to_gt + '/%03d.png' % i

            disp_est, img_est, gt = sess.run([model.disp, model.color_output, model.label_image])

            imgest = Image.fromarray(np.clip(img_est[0] * 255, 0, 255).astype(np.uint8))
            imggt = Image.fromarray(np.clip(gt[0] * 255, 0, 255).astype(np.uint8))
            imgdispest = Image.fromarray(np.clip((disp_est[0, :, :, 0] + 4) / 8 * 255, 0, 255).astype(np.uint8), mode='L')

            imgest.save(path_store_img_est)
            imggt.save(path_store_gt)
            imgdispest.save(path_store_disp_est)

            if i not in [0, 3, 6, 21, 24, 27, 42, 45, 48]:
                SSIM_val.append(measure.compare_ssim(np.clip(gt[0] * 255, 0, 255), np.clip(img_est[0] * 255, 0, 255),
                                                     data_range=255, multichannel=True, gaussian_weights=True))
                PSNR_val.append(measure.compare_psnr(np.clip(gt[0] * 255, 0, 255), np.clip(img_est[0] * 255, 0, 255),
                                                     data_range=255))
                MSE_val.append(measure.compare_mse(np.clip(gt[0] * 255, 0, 255), np.clip(img_est[0] * 255, 0, 255)))
                NRMSE_val.append(measure.compare_nrmse(np.clip(gt[0] * 255, 0, 255), np.clip(img_est[0] * 255, 0, 255)))
        ssim_mean = np.mean(SSIM_val)
        psnr_mean = np.mean(PSNR_val)
        mse_mean = np.mean(MSE_val)
        nrmse_mean = np.mean(NRMSE_val)
        with open(output_directory + '/' + args.model_name + '/psnr_ssim.txt', 'a') as fileobject:
            if num == 0:
                title = '{:<20}{:<30}{:<30}{:<30}{:<30}\n'.format('step' + str(times), 'SSIM', 'PSNR', 'MSE', 'NRMSE')
                fileobject.write(title)
            lines = '{:<20}{:<30}{:<30}{:<30}{:<30}\n'.format('Sample' + str(num) + ':', ssim_mean, psnr_mean,
                                                              mse_mean, nrmse_mean)
            fileobject.write(lines)

    print('calcul_done.')


def main(_):
    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary,
        num_gpus = args.num_gpus)

    if args.mode == 'train':
         train(params)
    elif args.mode == 'test':
         test(params)


if __name__ == '__main__':
    tf.app.run()





