"""Rain200L training/evaluation, CPU or CUDA. Run only on the target host."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DerainDataset, list_pairs, split_pairs, split_manifest
from models import build_model
from losses import VGGFeatures, PerceptualLoss, EasyContrastiveLoss
from runtime import select_device, seed_everything, restore_image, load_model
from utils import calculate_metrics, save_checkpoint, save_comparison


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--model', choices=['baseline','unet','derainnet','mymodel'], default='baseline')
    p.add_argument('--mode', choices=['train','test'], default='train')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--run_dir', type=Path, default=Path('runs/baseline'))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--patch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_fraction', type=float, default=.1)
    p.add_argument('--use_perceptual_loss', action='store_true')
    p.add_argument('--use_contrastive_loss', action='store_true')
    p.add_argument('--contrastive_mode', choices=['cosine','infonce'], default='cosine')
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--lambda_pixel', type=float, default=1.)
    p.add_argument('--lambda_perceptual', type=float, default=.1)
    p.add_argument('--lambda_contrastive', type=float, default=.1)
    p.add_argument('--lpips', action='store_true', help='Compute official AlexNet LPIPS (may download weights)')
    p.add_argument('--save_examples', type=int, default=3)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.patch_size < 32 or a.patch_size % 8 or a.num_workers < 0:
        p.error('epochs/batch positive, patch_size >=32 divisible by 8, workers >=0')
    if not math.isfinite(a.lr) or a.lr <= 0 or not math.isfinite(a.temperature) or a.temperature <= 0:
        p.error('learning rate and temperature must be finite positive numbers')
    if any(not math.isfinite(x) or x < 0 for x in (a.lambda_pixel,a.lambda_perceptual,a.lambda_contrastive)) or a.lambda_pixel == 0:
        p.error('loss weights finite nonnegative, lambda_pixel must be positive')
    if a.mode == 'train' and a.model != 'unet' and (a.use_perceptual_loss or a.use_contrastive_loss):
        p.error('architecture experiments must use MSE only; feature losses require the supplied unet')
    if a.save_examples < 0:
        p.error('save_examples must be nonnegative')
    if a.mode == 'test' and not a.checkpoint:
        p.error('test mode requires --checkpoint')
    if a.resume and a.resume.resolve().parent != a.run_dir.resolve():
        p.error('resume requires the checkpoint directory as run_dir; copy the whole run directory between machines')
    return a


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def json_write(path, value):
    def clean(x):
        if isinstance(x,float) and not math.isfinite(x):
            return str(x)
        if isinstance(x,dict):
            return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):
            return [clean(v) for v in x]
        return x
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def train_one_epoch(loader, model, optimizer, scaler, perceptual, contrastive, args, device):
    model.train()
    totals = dict(total=0.,pixel=0.,perceptual=0.,contrastive=0.)
    count = 0
    for rainy,clean,_ in tqdm(loader,desc='train'):
        rainy,clean = rainy.to(device,non_blocking=True),clean.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=args.amp and device.type=='cuda'):
            predicted = model(rainy)
            pixel = F.mse_loss(predicted,clean)
        # Feature distances in float32; freezing encoder parameters must not detach predictions.
        with torch.autocast(device_type=device.type,enabled=False):
            p_loss = perceptual(predicted.float(),clean) if perceptual is not None else pixel.new_zeros(())
            c_loss = contrastive(rainy,predicted.float(),clean) if contrastive is not None else pixel.new_zeros(())
            total = args.lambda_pixel*pixel + args.lambda_perceptual*p_loss + args.lambda_contrastive*c_loss
        if not torch.isfinite(total):
            raise FloatingPointError('nonfinite loss; reduce learning rate or disable AMP')
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer)
        scaler.update()
        n = rainy.shape[0]
        count += n
        for name,value in zip(totals,(total,pixel,p_loss,c_loss)):
            totals[name] += float(value.detach())*n
    return {k:v/count for k,v in totals.items()}


@torch.inference_mode()
def evaluate(loader, model, device, lpips_fn=None, output_dir=None, save_examples=0):
    model.eval()
    records = []
    for rainy,clean,names in tqdm(loader,desc='evaluate'):
        rainy,clean = rainy.to(device),clean.to(device)
        restored = restore_image(model,rainy).clamp(0,1)
        p,s,l = calculate_metrics(restored,clean,lpips_fn)
        records.append({'file':names[0],'psnr':p,'ssim':s,'lpips':l})
        if output_dir is not None and len(records) <= save_examples:
            save_comparison(rainy,restored,clean,Path(output_dir)/f'{Path(names[0]).stem}.png')
    if not records:
        raise ValueError('empty evaluation set')
    metrics = {key: (None if key=='lpips' and lpips_fn is None else float(np.mean([r[key] for r in records]))) for key in ('psnr','ssim','lpips')}
    return metrics, records


def main():
    args = get_args()
    device = select_device(args.device)
    seed_everything(args.seed)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    lpips_fn = None
    if args.lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    if args.mode == 'test':
        model,checkpoint = load_model(args.checkpoint,device)
        loader = DataLoader(DerainDataset(pairs=list_pairs(args.data_dir,'test'),is_train=False),batch_size=1,num_workers=args.num_workers)
        metrics,records = evaluate(loader,model,device,lpips_fn,args.run_dir/'test_images',args.save_examples)
        json_write(args.run_dir/'test_metrics.json',{'checkpoint':str(args.checkpoint),'model':checkpoint['model'],
                   'protocol':'RGB [0,1], full resolution, per-image mean, no border crop','metrics':metrics,'images':records})
        print(metrics)
        return
    if (args.run_dir/'last.pt').exists() and not args.resume:
        raise FileExistsError('run_dir already has last.pt; choose a new directory or --resume')
    train,val = split_pairs(args.data_dir,args.seed,args.val_fraction)
    manifest = split_manifest(train,val)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(DerainDataset(pairs=train,image_size=args.patch_size),batch_size=args.batch_size,
                              shuffle=True,num_workers=args.num_workers,pin_memory=device.type=='cuda',
                              worker_init_fn=worker_seed,generator=generator)
    val_loader = DataLoader(DerainDataset(pairs=val,is_train=False),batch_size=1,num_workers=args.num_workers)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda')
    perceptual = contrastive = None
    if args.use_perceptual_loss or args.use_contrastive_loss:
        encoder = VGGFeatures().to(device)
        perceptual = PerceptualLoss(encoder) if args.use_perceptual_loss else None
        contrastive = EasyContrastiveLoss(encoder,args.contrastive_mode,args.temperature) if args.use_contrastive_loss else None
    start,best = 0,-float('inf')
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.resume:
        checkpoint = torch.load(args.resume,map_location='cpu',weights_only=True)
        for key in ('model','epochs','batch_size','patch_size','seed','val_fraction','lr','amp','num_workers','use_perceptual_loss','use_contrastive_loss',
                    'contrastive_mode','temperature','lambda_pixel','lambda_perceptual','lambda_contrastive'):
            if checkpoint['args'][key] != config[key]:
                raise ValueError(f'resume configuration mismatch: {key}')
        if checkpoint['split']['sha256'] != manifest['sha256']:
            raise ValueError('train/val split changed')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start,best = checkpoint['epoch'],checkpoint['best_psnr']
        torch.set_rng_state(checkpoint['rng']['torch'])
        random.setstate(checkpoint['rng']['python'])
        state = checkpoint['rng']['numpy']
        np.random.set_state((state[0],np.asarray(state[1],dtype=np.uint32),state[2],state[3],state[4]))
        generator.set_state(checkpoint['rng']['loader'])
        if device.type=='cuda' and checkpoint['rng']['cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
    json_write(args.run_dir/'config.json',{'args':config,'split':manifest,'parameters':sum(p.numel() for p in model.parameters()),
               'torch':str(torch.__version__),'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None})
    fields = ['epoch','total','pixel','perceptual','contrastive','val_psnr','val_ssim','val_lpips','seconds','peak_cuda_mb','lr']
    log_path = args.run_dir/'metrics.csv'
    if not log_path.exists():
        with log_path.open('w',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=fields).writeheader()
    for epoch in range(start,args.epochs):
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        losses = train_one_epoch(train_loader,model,optimizer,scaler,perceptual,contrastive,args,device)
        metrics,_ = evaluate(val_loader,model,device,lpips_fn,args.run_dir/'val_images'/str(epoch+1),args.save_examples)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter()-begin
        improved = metrics['psnr'] > best
        best = max(best,metrics['psnr'])
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        np_state = np.random.get_state()
        rng = {'torch':torch.get_rng_state(),'python':random.getstate(),
               'numpy':[np_state[0],np_state[1].tolist(),np_state[2],np_state[3],np_state[4]],
               'loader':generator.get_state(),'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        checkpoint = {'model':args.model,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                      'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,
                      'best_psnr':best,'args':config,'split':manifest,'rng':rng}
        save_checkpoint(checkpoint,args.run_dir,'last.pt')
        if improved:
            save_checkpoint(checkpoint,args.run_dir,'best.pt')
        row = {'epoch':epoch+1,**losses,**{'val_'+k:v for k,v in metrics.items()},'seconds':seconds,
               'peak_cuda_mb':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0,'lr':lr}
        with log_path.open('a',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=fields).writerow(row)
        print(row,flush=True)


if __name__ == '__main__':
    main()
