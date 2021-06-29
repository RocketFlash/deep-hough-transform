import os
import torch

import yaml
from src.dataloader_new import get_loader
from src.model.network import Net
from inference import infer, get_model, load_weights
import cv2
import matplotlib.pyplot as plt



def main():
    torch.manual_seed(0)

    config_path = 'configs/config_retech_nestle_harr.yml'
    weights_path = 'work_dirs/harr_nestle_resnet50_fold0/model_best.pth'
    source_path = '/home/rauf/datasets/retechlabs/shelves_detection/harr/test_rotated/'
    save_path = 'work_dirs/harr_nestle_resnet50_fold0/model_traced.pt'
    
    # on_gpu = True
    on_gpu = False

    hough_cuda = False
    # hough_cuda = True

    assert os.path.isfile(config_path)
    CONFIGS = yaml.safe_load(open(config_path))
    device = torch.device('cuda:' + str(CONFIGS["TRAIN"]["GPU_ID"])) if on_gpu else torch.device("cpu")


    print('LOAD THE MODEL')
    model = get_model(num_angle=CONFIGS["MODEL"]["NUMANGLE"], 
                      num_rho=CONFIGS["MODEL"]["NUMRHO"], 
                      backbone=CONFIGS["MODEL"]["BACKBONE"], 
                      device=device,
                      hough_cuda=hough_cuda)

    
    checkpoint = load_weights(model, weights_path, device)
    print('MODEL LOADED')
        
    # dataloader
    test_loader = get_loader(source_path, None, batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"], test=True)
    sample, names = iter(test_loader).next()
    sample = sample.cuda(device=0) if on_gpu else sample
    print(model(sample))



    # image = cv2.imread('/home/rauf/datasets/retechlabs/shelves_detection/harr/test_rotated/51999.jpeg')
    # lines, ex_time = infer(image, model=model, 
    #                  input_size=(400,400), 
    #                  threshold=0.005, 
    #                  num_angle=100, 
    #                  num_rho=100,
    #                  show_time=True,
    #                  on_cuda=on_gpu)

    # image_with_lines = image.copy()
    # image_with_lines = cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB)
    # for line in lines:
    #     x1, y1, x2, y2 = [int(x) for x in line]
    #     cv2.line(image_with_lines,(x1,y1),(x2,y2),(0,0,255),9)

    # plt.imshow(image_with_lines)
    # plt.show()

    # traced_model = torch.jit.trace(model, sample)
    # print(traced_model.graph)
    # traced_model.save(save_path)


    # scripted_model = torch.jit.script(model)
    # print(scripted_model(sample))
    



if __name__ == '__main__':
    main()
