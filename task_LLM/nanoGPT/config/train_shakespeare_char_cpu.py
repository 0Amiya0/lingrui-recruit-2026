# train a miniature character-level shakespeare model on CPU
# (scaled down from train_shakespeare_char.py to run in reasonable time on CPU)

out_dir = 'out-shakespeare-char-cpu'
eval_interval = 150 # evaluate every 150 iterations
eval_iters = 25     # average over 25 batches for stable eval loss
log_interval = 10   # print loss every 10 iterations

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-cpu'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128 # context of up to 128 previous characters

# a smaller "baby GPT" so training is feasible on CPU
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2

learning_rate = 1e-3
max_iters = 1200
lr_decay_iters = 1200 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100

# CPU-only environment
device = 'cpu'
compile = False
dtype = 'float32'
