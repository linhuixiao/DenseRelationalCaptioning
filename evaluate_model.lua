require 'torch'
require 'nn'
require 'cudnn'

require 'densecap.DataLoader'
require 'densecap.DenseCapModel'

local utils = require 'densecap.utils'
local eval_utils = require 'eval.eval_utils_imRecall'

-- 参数配置
local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'checkpoint_VGlongv3_REM_tLSTM_MTL2_512_FC+nonlinear_1e6.t7',
  'The checkpoint to evaluate')
cmd:option('-data_h5', '', 'The HDF5 file to load data from; optional.')  -- 预处理好的数据
cmd:option('-data_json', '', 'The JSON file to load data from; optional.')  -- 预处理好的json文件
cmd:option('-gpu', 0, 'The GPU to use; set to -1 for CPU')
cmd:option('-use_cudnn', 1, 'Whether to use cuDNN backend in GPU mode.')
cmd:option('-split', 'test', 'Which split to evaluate; either val or test.')
cmd:option('-max_images', -1, 'How many images to evaluate; -1 for whole split')  -- 最多评估多少，默认全部
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.3)
cmd:option('-num_proposals', 50)
local opt = cmd:parse(arg)


-- 加载模型
-- First load the model
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
print 'Loaded model'

-- 设置是否使用CUDA，及数据类型
local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
print(string.format('Using dtype "%s"', dtype))

-- 将模型转移到相应的数据类型和cuda上来
model:convert(dtype, use_cudnn)
-- 设置测试参数，rpb nms阈值，proposal 数量
model:setTestArgs{
  rpn_nms_thresh=opt.rpn_nms_thresh,
  final_nms_thresh=opt.final_nms_thresh,
  max_proposals=opt.num_proposals,
}

-- 如果没有提供data_h5，data_json的路径，则使用cp中自带的路径
if opt.data_h5 == '' then
  opt.data_h5 = checkpoint.opt.data_h5
end
if opt.data_json == '' then
  opt.data_json = checkpoint.opt.data_json
end

-- 初始化和构建数据加载器
local loader = DataLoader(opt)

-- 构建参数列表
local eval_kwargs = {
  model=model,
  loader=loader,
  split=opt.split,
  max_images=opt.max_images,
  dtype=dtype,
}

-- 测试！
local eval_results = eval_utils.eval_split(eval_kwargs)

