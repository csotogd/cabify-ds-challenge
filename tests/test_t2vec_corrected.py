import unittest
from types import SimpleNamespace
import numpy as np
import torch
from routes.t2vec import EncoderDecoder,EOS
from routes.t2vec_corrected import full_spatial_loss


class FullSpatialLossTests(unittest.TestCase):
    def test_early_eos_is_penalized(self):
        torch.manual_seed(7);model=EncoderDecoder(6)
        vocab=SimpleNamespace(neighbors=np.arange(6)[:,None],weights=np.ones((6,1),np.float32))
        h=torch.randn(1,1,64);target=torch.tensor([[3]])
        before=full_spatial_loss(model,h,target,vocab)
        before.backward()
        self.assertGreater(float(model.output_bias.weight.grad[EOS]),0)
        with torch.no_grad():model.output_bias.weight[EOS]+=100
        after=full_spatial_loss(model,h,target,vocab)
        self.assertGreater(float(after.detach()),float(before.detach())+50)

    def test_single_target_equals_full_cross_entropy(self):
        torch.manual_seed(8);model=EncoderDecoder(6)
        vocab=SimpleNamespace(neighbors=np.arange(6)[:,None],weights=np.ones((6,1),np.float32))
        h=torch.randn(1,2,64);target=torch.tensor([[3,EOS]])
        logits=h.reshape(-1,64)@model.output.weight[2:].T+model.output_bias.weight[2:].flatten()
        expected=torch.nn.functional.cross_entropy(logits,target.flatten()-2)
        actual=full_spatial_loss(model,h,target,vocab)
        torch.testing.assert_close(actual,expected)

if __name__=='__main__':unittest.main()
