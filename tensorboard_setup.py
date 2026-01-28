from pathlib import Path
from torch.utils.tensorboard import SummaryWriter       #   tensorboard --logdir runs

# prepears tensorboard for logging LSTM
def tensorboard(log_dir='runs', run_name='test'):
    run_index = 1
    while (Path(log_dir) / f"{run_name}_stock_{run_index}").exists():
        run_index += 1
    log_path = f'{log_dir}/{run_name}_stock_{run_index}'
    writer = SummaryWriter(log_path)
    print(f"New run saved to: {log_path}")
    return writer