import torch

##########################################################################
# some postprocessing methods
#
# network output is a torch tensor of shape (batchsize, 4, h, w) before the softmax is applied.
# Therfore, this functions takes an input of the shape (batchsize, 4, h, w)
# it outputs a tensor of the shape (batchsize, 4, h, w) which is a one_hot representation of the input.
# reasoning: take one pixel-vector and find its argmax position. Convert to one_hot.
def network_output_to_one_hot(network_out):
  b = torch.argmax(network_out, dim=1)

  slice_0 = torch.unsqueeze((b == torch.zeros_like(b)), 1)
  slice_1 = torch.unsqueeze((b == torch.ones_like(b)), 1)
  slice_2 = torch.unsqueeze((b == 2 * torch.ones_like(b)), 1)
  slice_3 = torch.unsqueeze((b == 3 * torch.ones_like(b)), 1)

  c = torch.concat([slice_0, slice_1, slice_2, slice_3], dim = 1).long()
  return c

def out_to_one_hot(network_out, num_classes=4):
    b = torch.argmax(network_out, dim=1)

    slices = []
    for i in range(0, num_classes):
        slices.append(torch.unsqueeze((b == i * torch.ones_like(b)), 1))    

    return torch.concat(slices, dim = 1).long()

# input: tensor of shape (batchsize, 4, h, w)
# output: tensor of shape (batchsize, h, w) of floats normalized to [0., 1.]
def one_hot_to_greyscale(one_hot):
	return  (0. * one_hot[:,0,:,:] 
          + (1./3.) * one_hot[:,1,:,:] 
          + (2./3.) * one_hot[:,2,:,:] 
          + 1. * one_hot[:,3,:,:]
        )

if __name__ == '__main__':
    X = torch.rand(1, 2, 2, 2)
    
    # this can be used to test ones understanding of the code
    # Better with deterministic tensors instead of random ones.
    print(torch.squeeze(X))
    print(out_to_one_hot(X, 2))
    



