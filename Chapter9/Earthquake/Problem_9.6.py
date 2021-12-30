# -*- coding: utf-8 -*-

from BootstrapSize import BootstrapSize
from MovingBlock import MovingBlock
from ModelBased import ModelBased
from BlocksOfBlocks import BlocksOfBlocks

# =============================================================================
# # a.
# # m = 24
# model = BootstrapSize()
# block_size_mse = model.opt_mse()
# 
# # m = 25
# model_2 = BootstrapSize(B=10000, m=25, l_0=5)
# block_size_mse_2 = model_2.opt_mse(l_list=[1, 5, 25])
# 
# # b.
# model = MovingBlock()
# model.bootstrap_summary()
# 
# # c.
# model = ModelBased()
# model.bootstrap_summary()
# 
# =============================================================================
# d.
model = BlocksOfBlocks()
model.bootstrap_summary()