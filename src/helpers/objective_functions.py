### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import torch as tc
import torch.nn.functional as F

### Internal Imports ###
import ssim as s
import ssim_v2 as sv2

########################



def mean_absolute_error(tensor_1, tensor_2):
    return tc.mean(tc.abs(tensor_1 - tensor_2))

def mean_squared_error(tensor_1, tensor_2):
    return tc.mean((tensor_1 - tensor_2)**2)

def euclidean_distance(tensor_1, tensor_2):
    return tc.cdist(tensor_1.view(tensor_1.shape[0], -1), tensor_2.view(tensor_2.shape[0], -1))

def cosine_distance(tensor_1, tensor_2):
    return -tc.mean(tc.cosine_similarity(tensor_1.view(tensor_1.shape[0], -1), tensor_2.view(tensor_2.shape[0], -1)))

def pearson_correlation_coefficient(tensor_1, tensor_2):
    t1_mean = tc.mean(tensor_1)
    t2_mean = tc.mean(tensor_2)
    numerator = tc.sum((tensor_1 - t1_mean)*(tensor_2 - t2_mean))
    denominator = tc.linalg.norm(tensor_1 - t1_mean) * tc.linalg.norm(tensor_2 - t2_mean)
    return -numerator / (denominator + 1e-5)

def structural_similarity_index_measure(tensor_1, tensor_2):
    tensor_1 = (tensor_1 - tensor_1.min()) / (tensor_1.max() - tensor_1.min())
    tensor_2 = (tensor_2 - tensor_2.min()) / (tensor_2.max() - tensor_2.min())
    if tensor_1.shape[1] == 1:
        tensor_1 = tc.repeat_interleave(tensor_1, repeats=3, dim=1)
    if tensor_2.shape[1] == 1:
        tensor_2 = tc.repeat_interleave(tensor_2, repeats=3, dim=1)
    return -s.ssim3D(tensor_1, tensor_2)

def structural_similarity_index_measure_v2(tensor_1, tensor_2):
    tensor_1 = (tensor_1 - tensor_1.min()) / (tensor_1.max() - tensor_1.min())
    tensor_2 = (tensor_2 - tensor_2.min()) / (tensor_2.max() - tensor_2.min())
    return -sv2.SSIM(data_range=1.0, channel=1, spatial_dims=3)(tensor_1, tensor_2)






def test_ssim_3d():
    input_1 = tc.ones((1, 1, 256, 256, 256)) 
    # input_2 = tc.ones((1, 1, 256, 256, 256)) 
    input_2 = tc.zeros((1, 1, 256, 256, 256)) 
    input_2[:, :, 128:, 128:, 128:] = 1
    ssim = structural_similarity_index_measure(input_1, input_2)
    print(f"SSIM: {ssim}")


def run():
    test_ssim_3d()

if __name__ == "__main__":
    run()
