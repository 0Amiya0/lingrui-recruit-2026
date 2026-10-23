"""Sequential full experiment matrix; prints commands unless --execute is passed."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir',type=Path,required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--num_workers',type=int,default=0)
    p.add_argument('--amp',action='store_true')
    p.add_argument('--execute',action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    experiments = [('baseline',100,[]),('derainnet',50,[]),('mymodel',200,[]),('unet',100,[]),
                   ('unet',100,['--use_perceptual_loss']),('unet',100,['--use_contrastive_loss']),
                   ('unet',100,['--use_contrastive_loss','--contrastive_mode','infonce'])]
    for i,(model,epochs,extra) in enumerate(experiments):
        directory = root/'runs'/f'{i:02d}_{model}'
        command = [sys.executable,str(root/'main.py'),'--data_dir',str(a.data_dir.resolve()),'--model',model,
                   '--epochs',str(epochs),'--device',a.device,'--batch_size',str(a.batch_size),
                   '--num_workers',str(a.num_workers),'--run_dir',str(directory),*extra]
        if a.amp:
            command.append('--amp')
        print(subprocess.list2cmdline(command),flush=True)
        if a.execute:
            subprocess.run(command,cwd=root,check=True)
            subprocess.run([sys.executable,str(root/'main.py'),'--data_dir',str(a.data_dir.resolve()),'--mode','test',
                            '--checkpoint',str(directory/'best.pt'),'--device',a.device,'--lpips',
                            '--run_dir',str(directory)],cwd=root,check=True)


if __name__ == '__main__':
    main()
