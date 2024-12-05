import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
import json
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import time

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')

    parser.add_argument(
        '--exp-name',
        type=str,
        default='',
        help='Name to append to saved json file.')

    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')

    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of times to compute the provided image(s)')
    parser.add_argument('--save-output', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cpu', help='cpu or hpu')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args

def run_cam(args, model, target_layers, input_tensor, batch):
    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputTarget(281)]
    targets = None   

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    image_times = []

    for i in range(batch):
        start = time.time()
        with cam_algorithm(model=model,
                           target_layers=target_layers,
                           ) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            #end = time.time()
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


        gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
        gb = gb_model(input_tensor, target_category=None)

        end = time.time()
        image_times.append(end - start) 
    return image_times, grayscale_cam, gb


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()

    if args.device=='hpu':
        import habana_frameworks.torch.core as htcore

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM
    }

    model = models.resnet50(pretrained=True).to(torch.device(args.device)).eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    target_layers = [model.layer4]
    total_times = dict()
    #batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    batches = [1] 
    for batch in batches:
        print(batch)
        times_per_image = {}
        for fname in sorted(os.listdir(args.image_path)):
            rgb_img = cv2.imread(os.path.join(args.image_path, fname), 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]).to(args.device)
            
            # get all image times for this image
            image_times, grayscale_cam, gb = run_cam(args, model, target_layers, input_tensor, batch)
            print({fname: sum(image_times)})
            print()
            times_per_image[fname]= sum(image_times)
           
            if args.save_output:
                cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
                cam_gb = deprocess_image(cam_mask * gb)
                gb = deprocess_image(gb)

                os.makedirs(args.output_dir, exist_ok=True)

                cam_output_path = os.path.join(args.output_dir, f'{fname[:-5]}_{args.method}_cam.jpg')
                gb_output_path = os.path.join(args.output_dir, f'{fname[:-5]}_{args.method}_gb.jpg')
                cam_gb_output_path = os.path.join(args.output_dir, f'{fname[:-5]}_{args.method}_cam_gb.jpg')

                cv2.imwrite(cam_output_path, cam_image)
                cv2.imwrite(gb_output_path, gb)
                cv2.imwrite(cam_gb_output_path, cam_gb)
        print(times_per_image)
        total_times[batch] = times_per_image

    print(total_times) 

    # Save the dictionary to a file
    output_fname = f"{args.exp_name}_{args.method}_{args.device}_{batches[-1]}it_0"
    while os.path.exists(f'{output_fname}.json'):
        output_fname = output_fname[:-1] + str(int(output_fname[-1]) + 1)

    with open(f"{output_fname}.json", "w") as f:
        json.dump(total_times, f)
