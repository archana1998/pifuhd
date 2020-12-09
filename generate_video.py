from lib.colab_util import generate_video_from_obj, set_renderer, video
import torch
import os

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
              torch.cuda.empty_cache()
              generate_video_from_obj(obj_path, out_img_path, video_path, renderer)
              torch.cuda.empty_cache()

