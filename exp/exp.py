import torch
import numpy as np
import sys
import logging
from pathlib import Path
import PIL
import os

import dataset
import modules

sys.path.append("../")
import co
import ext
import co.utils as coutils
import config

import torch.multiprocessing as mp
from torch.multiprocessing import Process
from co.mytorch import dist, cleanup

import argparse


class Worker(co.mytorch.Worker):
    def __init__(
        self,
        train_dsets,
        eval_dsets="",
        train_n_nbs=1,
        train_nbs_mode="argmax",
        train_scale=1,
        train_patch=192,
        eval_n_nbs=1,
        eval_scale=-1,
        n_train_iters=750000,
        num_workers=8,
        **kwargs,
    ):
        super().__init__(
            n_train_iters=n_train_iters,
            num_workers=num_workers,
            train_device=config.train_device,
            eval_device=config.eval_device,
            **kwargs,
        )

        self.train_dsets = train_dsets
        self.eval_dsets = eval_dsets
        self.train_n_nbs = train_n_nbs
        self.train_nbs_mode = train_nbs_mode
        self.train_scale = train_scale
        self.train_patch = train_patch
        self.eval_n_nbs = eval_n_nbs
        self.eval_scale = train_scale if eval_scale <= 0 else eval_scale
        self.bwd_depth_thresh = 0.01
        self.invalid_depth_to_inf = True

        self.train_loss = modules.VGGPerceptualLoss()
        if config.lpips_root:
            self.eval_loss = modules.LPIPS()
        else:
            self.eval_loss = self.train_loss

    def get_pw_dataset(
        self,
        *,
        name,
        ibr_dir,
        im_size,
        patch,
        pad_width,
        n_nbs,
        nbs_mode,
        train,
        tgt_ind=None,
        n_max_sources=-1,
    ):
        logging.info(f"  create dataset for {name}")
        im_paths = sorted(ibr_dir.glob(f"im_*.png"))
        im_paths += sorted(ibr_dir.glob(f"im_*.jpg"))
        im_paths += sorted(ibr_dir.glob(f"im_*.jpeg"))
        dm_paths = sorted(ibr_dir.glob("dm_*.npy"))
        count_paths = sorted(ibr_dir.glob("count_*.npy"))
        counts = []
        for count_path in count_paths:
            counts.append(np.load(count_path))
        counts = np.array(counts)
        Ks = np.load(ibr_dir / "Ks.npy")
        Rs = np.load(ibr_dir / "Rs.npy")
        ts = np.load(ibr_dir / "ts.npy")

        if tgt_ind is None:
            tgt_ind = np.arange(len(im_paths))
            src_ind = np.arange(len(im_paths))
        else:
            src_ind = [
                idx for idx in range(len(im_paths)) if idx not in tgt_ind
            ]

        counts = counts[tgt_ind]
        counts = counts[:, src_ind]

        dset = dataset.Dataset(
            name=name,
            tgt_im_paths=[im_paths[idx] for idx in tgt_ind],
            tgt_dm_paths=[dm_paths[idx] for idx in tgt_ind],
            tgt_Ks=Ks[tgt_ind],
            tgt_Rs=Rs[tgt_ind],
            tgt_ts=ts[tgt_ind],
            tgt_counts=counts,
            src_im_paths=[im_paths[idx] for idx in src_ind],
            src_dm_paths=[dm_paths[idx] for idx in src_ind],
            src_Ks=Ks[src_ind],
            src_Rs=Rs[src_ind],
            src_ts=ts[src_ind],
            im_size=im_size,
            pad_width=pad_width,
            patch=patch,
            n_nbs=n_nbs,
            nbs_mode=nbs_mode,
            bwd_depth_thresh=self.bwd_depth_thresh,
            invalid_depth_to_inf=self.invalid_depth_to_inf,
            train=train,
        )
        return dset

    def get_track_dataset(
        self,
        name,
        src_ibr_dir,
        tgt_ibr_dir,
        n_nbs,
        im_size=None,
        pad_width=16,
        patch=None,
        nbs_mode="argmax",
        train=False,
    ):
        logging.info(f"  create dataset for {name}")

        src_im_paths = sorted(src_ibr_dir.glob(f"im_*.png"))
        src_im_paths += sorted(src_ibr_dir.glob(f"im_*.jpg"))
        src_im_paths += sorted(src_ibr_dir.glob(f"im_*.jpeg"))
        src_dm_paths = sorted(src_ibr_dir.glob("dm_*.npy"))
        src_Ks = np.load(src_ibr_dir / "Ks.npy")
        src_Rs = np.load(src_ibr_dir / "Rs.npy")
        src_ts = np.load(src_ibr_dir / "ts.npy")

        tgt_im_paths = sorted(tgt_ibr_dir.glob(f"im_*.png"))
        tgt_im_paths += sorted(tgt_ibr_dir.glob(f"im_*.jpg"))
        tgt_im_paths += sorted(tgt_ibr_dir.glob(f"im_*.jpeg"))
        if len(tgt_im_paths) == 0:
            tgt_im_paths = None
        tgt_dm_paths = sorted(tgt_ibr_dir.glob("dm_*.npy"))
        count_paths = sorted(tgt_ibr_dir.glob("count_*.npy"))
        counts = []
        for count_path in count_paths:
            counts.append(np.load(count_path))
        counts = np.array(counts)
        tgt_Ks = np.load(tgt_ibr_dir / "Ks.npy")
        tgt_Rs = np.load(tgt_ibr_dir / "Rs.npy")
        tgt_ts = np.load(tgt_ibr_dir / "ts.npy")

        dset = dataset.Dataset(
            name=name,
            tgt_im_paths=tgt_im_paths,
            tgt_dm_paths=tgt_dm_paths,
            tgt_Ks=tgt_Ks,
            tgt_Rs=tgt_Rs,
            tgt_ts=tgt_ts,
            tgt_counts=counts,
            src_im_paths=src_im_paths,
            src_dm_paths=src_dm_paths,
            src_Ks=src_Ks,
            src_Rs=src_Rs,
            src_ts=src_ts,
            im_size=im_size,
            pad_width=pad_width,
            patch=patch,
            n_nbs=n_nbs,
            nbs_mode=nbs_mode,
            bwd_depth_thresh=self.bwd_depth_thresh,
            invalid_depth_to_inf=self.invalid_depth_to_inf,
            train=train,
        )
        return dset

    def get_train_set_tat(self, dset):
        dense_dir = config.tat_root / dset / "dense"
        ibr_dir = dense_dir / f"ibr3d_pw_{self.train_scale:.2f}"
        dset = self.get_pw_dataset(
            name=f'tat_{dset.replace("/", "_")}',
            ibr_dir=ibr_dir,
            im_size=None,
            pad_width=16,
            patch=(self.train_patch, self.train_patch),
            n_nbs=self.train_n_nbs,
            nbs_mode=self.train_nbs_mode,
            train=True,
        )
        return dset

    def get_train_set_scannet(self, dset):
        _dir = config.scannet_root / "processed_train2" / dset
        num = len(list(_dir.glob('*.jpg')))
        start = int(num * 0.7)
        tgt_ind = [x for x in range(start, num)]
        
        dset = self.get_pw_dataset(
            name=f'scannet_{dset}',
            ibr_dir=_dir,
            im_size=None,
            pad_width=16,
            patch=(self.train_patch, self.train_patch),
            n_nbs=self.train_n_nbs,
            nbs_mode=self.train_nbs_mode,
            train=True,
            tgt_ind=tgt_ind
        )
        return dset

    def get_train_set(self):
        logging.info("Create train datasets")
        dsets = co.mytorch.MultiDataset(name="train")
        if "tat" in self.train_dsets:
            for dset in config.tat_train_sets:
                dsets.append(self.get_train_set_tat(dset))
        if "scannet" in self.train_dsets:
            for dset in config.scannet_train_sets:
                dsets.append(self.get_train_set_scannet(dset))
        return dsets

    def get_eval_set_tat(self, dset, mode):
        dense_dir = config.tat_root / dset / "dense"
        ibr_dir = dense_dir / f"ibr3d_pw_{self.eval_scale:.2f}"
        if mode == "all":
            tgt_ind = None
        elif mode == "subseq":
            tgt_ind = config.tat_eval_tracks[dset]
        else:
            raise Exception("invalid mode for get_eval_set_tat")
        dset = self.get_pw_dataset(
            name=f'tat_{mode}_{dset.replace("/", "_")}',
            ibr_dir=ibr_dir,
            im_size=None,
            pad_width=16,
            patch=None,
            n_nbs=self.eval_n_nbs,
            nbs_mode="argmax",
            tgt_ind=tgt_ind,
            train=False,
        )
        return dset
    
    def get_eval_set_scannet(self, dset, mode):
        _dir = config.scannet_root / "processed_test2" / dset
        if mode == "all":
            tgt_ind = None
        elif mode == "subseq":
            num = len(list(_dir.glob('*.jpg')))
            start = int(num * 0.7)
            tgt_ind = [x for x in range(start, num)]
        dset = self.get_pw_dataset(
            name=f'scannet_{dset}',
            ibr_dir=_dir,
            im_size=None,
            pad_width=16,
            patch=None,
            n_nbs=self.eval_n_nbs,
            nbs_mode="argmax",
            train=False,
            tgt_ind=tgt_ind
        )
        return dset

    def get_eval_sets(self):
        logging.info("Create eval datasets")
        eval_sets = []

        if "scannet" in self.eval_dsets:
            for dset in config.scannet_eval_sets:
                dset = self.get_eval_set_scannet(dset, "subseq")
                eval_sets.append(dset)

        if "tat" in self.eval_dsets:
            for dset in config.tat_eval_sets:
                dset = self.get_eval_set_tat(dset, "all")
                eval_sets.append(dset)
        for dset in self.eval_dsets:
            if dset.startswith("tat-scene-"):
                dset = dset[len("tat-scene-") :]
                dset = self.get_eval_set_tat(dset, "all")
                eval_sets.append(dset)
        if "tat-subseq" in self.eval_dsets:
            for dset in config.tat_eval_sets:
                dset = self.get_eval_set_tat(dset, "subseq")
                eval_sets.append(dset)
        for dset in eval_sets:
            dset.logging_rate = 1
            dset.vis_ind = np.arange(len(dset))
        return eval_sets

    def copy_data(self, data, device, train):
        self.data = {}
        for k, v in data.items():
            self.data[k] = v.to(device).requires_grad_(requires_grad=False)

    def net_forward(self, net, train, iter):
        return net(**self.data)

    def loss_forward(self, output, train, iter):
        errs = {}
        tgt = self.data["tgt"]
        est = output["out"]
        est = est[..., : tgt.shape[-2], : tgt.shape[-1]]

        if train:
            for lidx, loss in enumerate(self.train_loss(est, tgt)):
                errs[f"rgb{lidx}"] = loss
        else:
            est = torch.clamp(est, -1, 1)
            est = 255 * (est + 1) / 2
            est = est.type(torch.uint8)
            est = est.type(torch.float32)
            est = (est / 255 * 2) - 1
            errs["rgb"] = self.eval_loss(est, tgt)

        output["out"] = est

        return errs

    def callback_eval_start(self, **kwargs):
        self.metric = None

    def im_to2np(self, im):
        im = im.detach().to("cpu").numpy()
        im = (np.clip(im, -1, 1) + 1) / 2
        im = im.transpose(0, 2, 3, 1)
        return im

    def callback_eval_add(self, **kwargs):
        output = kwargs["output"]
        batch_idx = kwargs["batch_idx"]
        iter = kwargs["iter"]
        eval_set = kwargs["eval_set"]
        eval_set_name = eval_set.name.replace("/", "_")
        eval_set_name = f"{eval_set_name}_{self.eval_scale}"

        ta = self.im_to2np(self.data["tgt"])
        es = self.im_to2np(output["out"])

        # record metrics
        if self.metric is None:
            self.metric = {}
            self.metric["rgb"] = co.metric.MultipleMetric(
                metrics=[
                    co.metric.DistanceMetric(p=1, vec_length=3),
                    co.metric.PSNRMetric(),
                    co.metric.SSIMMetric(),
                ]
            )

        self.metric["rgb"].add(es, ta)

        # write debug images
        out_dir = self.exp_out_root / f"{eval_set_name}_n{self.eval_n_nbs}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in range(ta.shape[0]):
            bidx = batch_idx * ta.shape[0] + b
            if bidx not in eval_set.vis_ind:
                continue

            tgt_dm = self.data["tgt_dm"][b].detach().to("cpu").numpy()

            out_im = (255 * es[b]).astype(np.uint8)
            if hasattr(eval_set, "mask_via_depth") and eval_set.mask_via_depth:
                out_im[tgt_dm <= 0] = 255
                out_im[tgt_dm >= 1e6] = 255
            # PIL.Image.fromarray(out_im).save(out_dir / f"{bidx:04d}_es.png")
            PIL.Image.fromarray(out_im).save(out_dir / f"s{bidx:04d}_es.jpg")
            out_im = (255 * ta[b]).astype(np.uint8)
            PIL.Image.fromarray(out_im).save(out_dir / f"{bidx:04d}_ta.png")

    def callback_eval_stop(self, **kwargs):
        eval_set = kwargs["eval_set"]
        iter = kwargs["iter"]
        mean_loss = kwargs["mean_loss"]
        eval_set_name = eval_set.name.replace("/", "_")
        eval_set_name = f"{eval_set_name}_{self.eval_scale}"
        method = self.experiment_name + f"_n{self.eval_n_nbs}"
        for key in self.metric:
            self.metric_add_eval(
                iter,
                eval_set_name,
                f"loss_{key}",
                sum(np.asarray(mean_loss[key]).ravel()),
                method=method,
            )
            metric = self.metric[key]
            logging.info(f"\n{key}\n{metric}")
            for k, v in metric.items():
                self.metric_add_eval(
                    iter, eval_set_name, f"{k}", v, method=method
                )


def main_func(rank, world_size, args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    args.rank = rank
    args.world_size = world_size
    args.gpu = rank

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

    rank = args.rank
    device = torch.device(args.device)

    experiment_name = f"{'+'.join(args.train_dsets)}_nbs{args.train_n_nbs}_s{args.train_scale}_p{args.train_patch}_{args.net}"
    worker = Worker(
        # distributed
        rank=rank,
        device=device,
        world_size=args.world_size,
        # 
        experiments_root=args.experiments_root,
        experiment_name=experiment_name,
        train_dsets=args.train_dsets,
        eval_dsets=args.eval_dsets,
        train_n_nbs=args.train_n_nbs,
        train_scale=args.train_scale,
        train_patch=args.train_patch,
        eval_n_nbs=args.eval_n_nbs,
        eval_scale=args.eval_scale,
        n_train_iters=750000,
    )
    worker.log_debug = args.log_debug
    
    if rank == 0:
        worker.save_frequency = co.mytorch.Frequency(hours=1)
        worker.eval_frequency = co.mytorch.Frequency(hours=1)

    worker.train_batch_size = 1
    worker.eval_batch_size = 1
    worker.train_batch_acc_steps = 1

    worker_objects = co.mytorch.WorkerObjects(
        optim_f=lambda net: torch.optim.Adam(net.parameters(), lr=1e-4 * world_size)
    )

    worker_objects.net_f = lambda: modules.get_rnn_net(
            enc_net="vgg16unet3", merge_net="gruunet4.64.3"
    )

    worker.do(args, worker_objects)

    cleanup()


if __name__ == "__main__":
    commands = ["retrain", "resume", "eval", "eval-init", "slurm"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default="resume", choices=commands)
    parser.add_argument("--log-env-info", type=coutils.str2bool, default=False)
    parser.add_argument("--iter", type=str, nargs="*", default=[])
    parser.add_argument("--eval-net-root", type=str, default="")
    parser.add_argument("--experiments-root", type=str, default="./experiments_scannet")
    parser.add_argument("--slurm-cmd", type=str, default="resume")
    parser.add_argument("--slurm-queue", type=str, default="gpu")
    parser.add_argument("--slurm-n-gpus", type=int, default=1)
    parser.add_argument("--slurm-n-cpus", type=int, default=-1)
    parser.add_argument(
        "--slurm-time",
        type=str,
        default="2-00:00",
        help='Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"',
    )
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--train-dsets", nargs="+", type=str, default=["tat"])
    #parser.add_argument("--train-dsets", nargs="+", type=str, default=["scannet"])
    #parser.add_argument(
    #    "--eval-dsets", nargs="+", type=str, default=["scannet"]
    #)
    parser.add_argument(
        "--eval-dsets", nargs="+", type=str, default=["tat", "tat-subseq"]
    )
    parser.add_argument("--train-n-nbs", type=int, default=5)
    parser.add_argument("--train-scale", type=float, default=0.25)
    parser.add_argument("--train-patch", type=int, default=192)
    parser.add_argument("--eval-n-nbs", type=int, default=5)
    parser.add_argument("--eval-scale", type=float, default=-1)
    parser.add_argument("--log-debug", type=str, nargs="*", default=[])
    # initialization multi gpu
    parser.add_argument("--world_size", default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device')

    args = parser.parse_args()

    world_size = args.world_size
    processes = []
    for rank in range(world_size):
        p = Process(target=main_func, args=(rank, world_size, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

