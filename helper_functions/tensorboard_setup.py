from pathlib import Path
from torch.utils.tensorboard import SummaryWriter       #   tensorboard --logdir runs

# prepears tensorboard for logging LSTM
def tensorboard(log_dir='runs/training', run_name='test', custom=""):
    run_index = 1
    while (Path(log_dir) / f"{run_name}_stock_{run_index}_{custom}").exists():
        run_index += 1

    name = f"{run_name}_stock_{run_index}_{custom}"
    log_path = f'{log_dir}/{name}_{custom}_training'

    writer = SummaryWriter(log_path)
    
    print(f"New run saved to: {log_path}")
    return writer, name