from .DVLDE import DVLDE

def get_model(opt):
  if opt.model == 'basic':
    model = DVLDE(opt)
  else:
    raise NotImplementedError

  model.setup_training()
  model.initialize_weights()
  return model
