重新在队列中挂起两个实验

python3 train_ve_ri_relu2_gate_speedrun.py config/train_gpt2.py \
  --batch_size=24 --gradient_accumulation_steps=10 --warmup_iters=100 \
  --learning_rate=1.8e-3 --min_lr=1e-6 --lr_schedule=cosine \
  --muon_weight_decay=0.01 --speedrun_mode=True \
  --speedrun_time_limit_seconds=14400 \
  --wandb_run_name='crz dense-relu2-gate-speedrun-4h' \
  --out_dir=out-dense-relu2-gate-speedrun \
  > /tmp/exp_dense_speedrun.log 2>&1

python3 train_ve_ri_gate_moe_speedrun.py config/train_gpt2.py \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --warmup_iters=100 \
  --learning_rate=1e-3 --muon_lr=1e-2 --min_lr=1e-6 \
  --lr_schedule=wsd --wsd_decay_frac=0.3 --wsd_decay_style=linear \
  --muon_weight_decay=0.01 \
  --moe_activation=swiglu --moe_expert_dim=1024 \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --wandb_run_name='crz moe-all-swiglu-d1024-wsd-speedrun-4h' \
  --out_dir=out-moe-swiglu-wsd-speedrun \
  > /tmp/exp_moe_swiglu_wsd_speedrun.log 2>&1

  python3 train_ve_ri_gate_latent_moe_speedrun.py \
  config/train_gpt2.py \
  --batch_size=24 \
  --gradient_accumulation_steps=10 \
  --warmup_iters=100 \
  --learning_rate=1e-3 --muon_lr=1e-2 --min_lr=1e-6 \
  --lr_schedule=wsd --wsd_decay_frac=0.3 --wsd_decay_style=linear \
  --muon_weight_decay=0.01 \
  --speedrun_mode=True \
  --speedrun_time_limit_seconds=14400 \
  --wandb_run_name='crz latent-moe-eff-a2-single-4h' \
  --out_dir=out-latent-moe-eff-a2-single \
  > /tmp/exp_latent_moe_single.log 2>&1



我现在希望做两个实验，依次启动：将ve_ri_gate_moe系列代码中激活函数换成swiglue（总参数量保持不变），将其挂起为第一个实验；然后将expert_dim/2（参数量减半，swiglue部分对应减半），运行第二个实验。