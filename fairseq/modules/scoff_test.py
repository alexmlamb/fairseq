

'''
                nhid,
                num_blocks_in,
                num_blocks_out,
                topkval,
                memorytopk,
                step_att,
                num_modules_read_input,
                inp_heads,
                do_gru,
                do_rel,
                n_templates,
                share_inp,
                share_comm,
                memory_slots=4,
                num_memory_heads=4,
                memory_head_size=16,
                memory_mlp=4,
                attention_out=340,
                version=1,
                device=None,

'''
import torch
from scoff_blocks_core_bb import BlocksCore

bc = BlocksCore(nhid_in = 512*3, nhid=512, num_blocks_in=3, num_blocks_out=1, topkval=1, memorytopk=1, step_att=True, num_modules_read_input=1, inp_heads=1, do_gru=True, do_rel=False, n_templates=2, share_inp=False, share_comm=False, memory_slots=1, num_memory_heads=1, memory_head_size=1, memory_mlp=1, attention_out=512, version=1)

bc.blockify_params()

x = torch.randn(1,3*512)
h = torch.randn(1,512)

h,_,_,_,_ = bc(x,h)

print('h shape out', h.shape)


