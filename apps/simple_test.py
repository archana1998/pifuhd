# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from .recon import reconWrapper
import argparse
import torch
import os

torch.cuda.empty_cache()
###############################################################################################
##                   Setting
###############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str, default='./sample_images')
parser.add_argument('-o', '--out_path', type=str, default='./results')
parser.add_argument('-c', '--ckpt_path', type=str, default='./checkpoints/pifuhd.pt')
parser.add_argument('-r', '--resolution', type=int, default=512)
parser.add_argument('--use_rect', action='store_true', help='use rectangle for cropping')
args = parser.parse_args()


###############################################################################################
##                   Upper PIFu
###############################################################################################

resolution = str(args.resolution)

start_id = -1
end_id = -1
cmd = ['--dataroot', args.input_path, '--results_path', args.out_path,\
       '--loadSize', '1024', '--resolution', resolution, '--load_netMR_checkpoint_path', \
       args.ckpt_path,\
       '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]
reconWrapper(cmd, args.use_rect)

from lib.colab_util import generate_video_from_obj, set_renderer, video

renderer = set_renderer()
for filename in os.listdir("/home/archana/anaconda3/pifuhd/sample_images"):
       if filename.endswith(".jpg") or filename.endswith("png"):
              image_path = '/home/archana/anaconda3/pifuhd/sample_images/%s' % filename
              image_dir = os.path.dirname(image_path)
              file_name = os.path.splitext(os.path.basename(image_path))[0]

              # output pathes
              obj_path = '/home/archana/anaconda3/pifuhd/results/pifuhd_final/recon/result_%s_512.obj' % file_name
              out_img_path = '/home/archana/anaconda3/pifuhd/results/pifuhd_final/recon/result_%s_512.png' % file_name
              video_path = '/home/archana/anaconda3/pifuhd/results/pifuhd_final/recon/result_%s_512.mp4' % file_name
              video_display_path = '/home/archana/anaconda3/pifuhd/results/pifuhd_final/result_%s_512_display.mp4' % file_name
              generate_video_from_obj(obj_path, out_img_path, video_path, renderer)

