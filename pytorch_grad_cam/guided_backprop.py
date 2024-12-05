import numpy as np
import torch
from torch.autograd import Function
from pytorch_grad_cam.utils.find_layers import replace_all_layer_type_recursive
from time import time


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(
            torch.zeros(
                input_img.size()).type_as(input_img),
            input_img,
            positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(
                input_img.size()).type_as(input_img),
            torch.addcmul(
                torch.zeros(
                    input_img.size()).type_as(input_img),
                grad_output,
                positive_mask_1),
            positive_mask_2)
        return grad_input


class GuidedBackpropReLUasModule(torch.nn.Module):
    def __init__(self):
        super(GuidedBackpropReLUasModule, self).__init__()

    def forward(self, input_img):
        return GuidedBackpropReLU.apply(input_img)


class GuidedBackpropReLUModel:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def forward(self, input_img):
        return self.model(input_img)

    def recursive_replace_relu_with_guidedrelu(self, module_top):

        for idx, module in module_top._modules.items():
            self.recursive_replace_relu_with_guidedrelu(module)
            if module.__class__.__name__ == 'ReLU':
                module_top._modules[idx] = GuidedBackpropReLU.apply
        print("b")

    def recursive_replace_guidedrelu_with_relu(self, module_top):
        try:
            for idx, module in module_top._modules.items():
                self.recursive_replace_guidedrelu_with_relu(module)
                if module == GuidedBackpropReLU.apply:
                    module_top._modules[idx] = torch.nn.ReLU()
        except BaseException:
            pass

    def __call__(self, input_img, target_category=None):
        replace_all_layer_type_recursive(self.model,
                                         torch.nn.ReLU,
                                         GuidedBackpropReLUasModule())


        input_img = input_img.to(self.device)

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img).to(self.device)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        loss = output[0, target_category]
        loss.backward(retain_graph=True)

        '''
        START OF POC CODE TO ACCELERATE ON GAUDI
        '''
        hpu = 'hpu' in str(self.device)
        if hpu:
            import habana_frameworks.torch.core as htcore
            import habana_frameworks.torch as htorch
            htcore.mark_step()

        
        start = time()
        if hpu: 
            with hpu.metrics.metric_localcontext("graph_compilation") as local_metric:
                output = input_img.grad.data
        else:
            output = input_img.grad.cpu().data.numpy()
        end = time()
        if hpu:
            print(htorch.hpu.memory_summary())
            print(local_metric.stats())
        print('autograd time:',end-start)

        output = output[0, :, :, :]
        start = time()
        if hpu:
            output = output.permute(1, 2, 0)
        else:
            output = output.transpose((1, 2, 0))
        end = time()
        print('Transpose time:',end-start)
        '''
        END OF POC CODE TO ACCELERATE ON GAUDI
        '''
        replace_all_layer_type_recursive(self.model,
                                         GuidedBackpropReLUasModule,
                                         torch.nn.ReLU())

        return output
