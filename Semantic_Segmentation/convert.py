import torch
import ever as er
er.registry.register_modules()
config_path='baseline.hrnetw32'
checkpoint_path="./log/hrnetw32.pth"

model, gs = er.infer_tool.build_and_load_from_file(config_path, checkpoint_path)


torch.save(model.state_dict(),'./log/hrnetw32-.pth')