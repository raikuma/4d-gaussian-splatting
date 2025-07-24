CUDA_VISIBLE_DEVICES=2 python train.py --config configs/dynerf_less/coffee_martini.yaml --num_workers 6 &&
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/dynerf_less/cook_spinach.yaml --num_workers 6 &&
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/dynerf_less/cut_roasted_beef.yaml --num_workers 6
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/dynerf_less/flame_salmon.yaml --num_workers 6 &&
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/dynerf_less/flame_steak.yaml --num_workers 6 &&
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/dynerf_less/sear_steak.yaml --num_workers 6

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/dynerf/coffee_martini.yaml --num_workers 6 &&
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/dynerf/cook_spinach.yaml --num_workers 6 &&
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/dynerf/cut_roasted_beef.yaml --num_workers 6
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/dynerf/flame_salmon.yaml --num_workers 6 &&
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/dynerf/flame_steak.yaml --num_workers 6 &&
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/dynerf/sear_steak.yaml --num_workers 6

CUDA_VISIBLE_DEVICES=3 python train.py --config configs/technicolor_15K/birthday.yaml --num_workers 6 ;
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/technicolor_15K/fabien.yaml --num_workers 6 ;
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/technicolor_15K/painter.yaml --num_workers 6 ;
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/technicolor_15K/theater.yaml --num_workers 6 ;
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/technicolor_15K/train.yaml --num_workers 6

CUDA_VISIBLE_DEVICES=2 sh metric_all.sh output/technicolor_50_15K/Birthday data/technicolor_50/Birthday/combined_motion_masks 15000;
CUDA_VISIBLE_DEVICES=2 sh metric_all.sh output/technicolor_50_15K/Theater data/technicolor_50/Theater/combined_motion_masks 15000;
CUDA_VISIBLE_DEVICES=2 sh metric_all.sh output/technicolor_50_15K/Fabien data/technicolor_50/Fabien/combined_motion_masks 15000;
CUDA_VISIBLE_DEVICES=2 sh metric_all.sh output/technicolor_50_15K/Painter data/technicolor_50/Painter/combined_motion_masks 15000;
CUDA_VISIBLE_DEVICES=2 sh metric_all.sh output/technicolor_50_15K/Train data/technicolor_50/Train/combined_motion_masks 15000

python metric_number.py -m /mnt/d/files/iclr_4dgs_dynerf_log/coffee_martini_wo_cam13/ -s /mnt/d/data/N3DV/coffee_martini/ --loaded_pth /mnt/d/files/iclr_4dgs_dynerf_log/coffee_martini_wo_cam13/chkpnt_best.pth ;
python metric_number.py -m /mnt/d/files/iclr_4dgs_dynerf_log/cook_spinach/ -s /mnt/d/data/N3DV/cook_spinach/ --loaded_pth /mnt/d/files/iclr_4dgs_dynerf_log/cook_spinach/chkpnt_best.pth ;
python metric_number.py -m /mnt/d/files/iclr_4dgs_dynerf_log/cut_roasted_beef/ -s /mnt/d/data/N3DV/cut_roasted_beef/ --loaded_pth /mnt/d/files/iclr_4dgs_dynerf_log/cut_roasted_beef/chkpnt_best.pth ;
python metric_number.py -m /mnt/d/files/iclr_4dgs_dynerf_log/flame_salmon_0/ -s /mnt/d/data/N3DV/flame_salmon_1/ --loaded_pth /mnt/d/files/iclr_4dgs_dynerf_log/flame_salmon_0/chkpnt_best.pth ;
python metric_number.py -m /mnt/d/files/iclr_4dgs_dynerf_log/flame_steak/ -s /mnt/d/data/N3DV/flame_steak/ --loaded_pth /mnt/d/files/iclr_4dgs_dynerf_log/flame_steak/chkpnt_best.pth ;
python metric_number.py -m /mnt/d/files/iclr_4dgs_dynerf_log/sear_steak/ -s /mnt/d/data/N3DV/sear_steak/ --loaded_pth /mnt/d/files/iclr_4dgs_dynerf_log/sear_steak/chkpnt_best.pth ;

sh metric_all.sh /mnt/d/files/iclr_4dgs_dynerf_log/coffee_martini_wo_cam13/ data/N3DV/coffee_martini/combined_motion_masks None ;
sh metric_all.sh /mnt/d/files/iclr_4dgs_dynerf_log/cook_spinach/ data/N3DV/cook_spinach/combined_motion_masks None ;
sh metric_all.sh /mnt/d/files/iclr_4dgs_dynerf_log/cut_roasted_beef/ data/N3DV/cut_roasted_beef/combined_motion_masks None ;
sh metric_all.sh /mnt/d/files/iclr_4dgs_dynerf_log/bin/flame_salmon_0/ data/N3DV/flame_salmon_1/combined_motion_masks None ;
sh metric_all.sh /mnt/d/files/iclr_4dgs_dynerf_log/flame_steak/ data/N3DV/flame_steak/combined_motion_masks None ;
sh metric_all.sh /mnt/d/files/iclr_4dgs_dynerf_log/sear_steak/ data/N3DV/sear_steak/combined_motion_masks None

CUDA_VISIBLE_DEVICES=0 sh metric_all.sh output/N3V/coffee_martini/ data/N3V/coffee_martini/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=0 sh metric_all.sh output/N3V/cook_spinach/ data/N3V/cook_spinach/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=0 sh metric_all.sh output/N3V/cut_roasted_beef/ data/N3V/cut_roasted_beef/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=0 sh metric_all.sh output/N3V/flame_salmon_1/ data/N3V/flame_salmon_1/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=0 sh metric_all.sh output/N3V/flame_steak/ data/N3V/flame_steak/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=0 sh metric_all.sh output/N3V/sear_steak/ data/N3V/sear_steak/combined_motion_masks/ 30000 ;

CUDA_VISIBLE_DEVICES=1 sh metric_all.sh output/N3V_da/coffee_martini/ data/N3V/coffee_martini/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=1 sh metric_all.sh output/N3V_da/cook_spinach/ data/N3V/cook_spinach/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=1 sh metric_all.sh output/N3V_da/cut_roasted_beef/ data/N3V/cut_roasted_beef/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=1 sh metric_all.sh output/N3V_da/flame_salmon_1/ data/N3V/flame_salmon_1/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=1 sh metric_all.sh output/N3V_da/flame_steak/ data/N3V/flame_steak/combined_motion_masks/ 30000 ;
CUDA_VISIBLE_DEVICES=1 sh metric_all.sh output/N3V_da/sear_steak/ data/N3V/sear_steak/combined_motion_masks/ 30000

sh metric_all.sh output/N3V_less/coffee_martini/ data/N3V/coffee_martini/combined_motion_masks/ 30000 ;
sh metric_all.sh output/N3V_less/cook_spinach/ data/N3V/cook_spinach/combined_motion_masks/ 30000 ;
sh metric_all.sh output/N3V_less/cut_roasted_beef/ data/N3V/cut_roasted_beef/combined_motion_masks/ 30000 ;
sh metric_all.sh output/N3V_less/flame_salmon_1/ data/N3V/flame_salmon_1/combined_motion_masks/ 30000 ;
sh metric_all.sh output/N3V_less/flame_steak/ data/N3V/flame_steak/combined_motion_masks/ 30000 ;
sh metric_all.sh output/N3V_less/sear_steak/ data/N3V/sear_steak/combined_motion_masks/ 30000

sh metric_all.sh /mnt/d/results/4dgs_technicolor/Birthday/ data/technicolor_15K/Birthday/combined_motion_masks 30000 ;
sh metric_all.sh /mnt/d/results/4dgs_technicolor/Fabien/ data/technicolor_15K/Fabien/combined_motion_masks 30000 ;
sh metric_all.sh /mnt/d/results/4dgs_technicolor/Painter/ data/technicolor_15K/Painter/combined_motion_masks 30000 ;
sh metric_all.sh /mnt/d/results/4dgs_technicolor/Theater/ data/technicolor_15K/Theater/combined_motion_masks 30000 ;
sh metric_all.sh /mnt/d/results/4dgs_technicolor/Train/ data/technicolor_15K/Train/combined_motion_masks 30000

python metric_number.py -m /mnt/d/results/4dgs_technicolor/Birthday/ -s /mnt/d/data/technicolor_50/Birthday/ --loaded_pth /mnt/d/results/4dgs_technicolor/Birthday/chkpnt_best.pth ;
python metric_number.py -m /mnt/d/results/4dgs_technicolor/Fabien/ -s /mnt/d/data/technicolor_50/Fabien/ --loaded_pth /mnt/d/results/4dgs_technicolor/Fabien/chkpnt_best.pth ;
python metric_number.py -m /mnt/d/results/4dgs_technicolor/Painter/ -s /mnt/d/data/technicolor_50/Painter/ --loaded_pth /mnt/d/results/4dgs_technicolor/Painter/chkpnt_best.pth ;
python metric_number.py -m /mnt/d/results/4dgs_technicolor/Theater/ -s /mnt/d/data/technicolor_50/Theater/ --loaded_pth /mnt/d/results/4dgs_technicolor/Theater/chkpnt_best.pth ;
python metric_number.py -m /mnt/d/results/4dgs_technicolor/Train/ -s /mnt/d/data/technicolor_50/Train/ --loaded_pth /mnt/d/results/4dgs_technicolor/Train/chkpnt_best.pth ;

python metric_number.py -m output/technicolor_50_15K/Fabien/ --skip_train --loaded_pth output/technicolor_50_15K/Fabien/chkpnt15000.pth
python metric_number.py -m output/technicolor_50_15K/Painter/ --skip_train --loaded_pth output/technicolor_50_15K/Painter/chkpnt15000.pth
python metric_number.py -m output/technicolor_50_15K/Train/ --skip_train --loaded_pth output/technicolor_50_15K/Train/chkpnt15000.pth

python train.py --config configs/technicolor_22K/birthday.yaml --num_workers 6 &&
python train.py --config configs/technicolor_22K/fabien.yaml --num_workers 6 &&
python train.py --config configs/technicolor_22K/painter.yaml --num_workers 6 &&
python train.py --config configs/technicolor_22K/theater.yaml --num_workers 6 &&
python train.py --config configs/technicolor_22K/train.yaml --num_workers 6

sh metric_all.sh output/technicolor_50_22K/Birthday/ data/technicolor_50/Birthday/combined_motion_masks 22500 &&
sh metric_all.sh output/technicolor_50_22K/Fabien/ data/technicolor_50/Fabien/combined_motion_masks 22500 &&
sh metric_all.sh output/technicolor_50_22K/Painter/ data/technicolor_50/Painter/combined_motion_masks 22500 &&
sh metric_all.sh output/technicolor_50_22K/Theater/ data/technicolor_50/Theater/combined_motion_masks 22500 &&
sh metric_all.sh output/technicolor_50_22K/Train/ data/technicolor_50/Train/combined_motion_masks 22500

python train.py --config configs/dynerf/coffee_martini.yaml --num_workers 6 --ply_path data/N3V/coffee_martini/colmap_0/sparse/0/points3D.ply &&
python train.py --config configs/dynerf/cook_spinach.yaml --num_workers 6 --ply_path data/N3V/cook_spinach/colmap_0/sparse/0/points3D.ply &&
python train.py --config configs/dynerf/cut_roasted_beef.yaml --num_workers 6 --ply_path data/N3V/cut_roasted_beef/colmap_0/sparse/0/points3D.ply &&
python train.py --config configs/dynerf/flame_salmon.yaml --num_workers 6 --ply_path data/N3V/flame_salmon_1/colmap_0/sparse/0/points3D.ply &&
python train.py --config configs/dynerf/flame_steak.yaml --num_workers 6 --ply_path data/N3V/flame_steak/colmap_0/sparse/0/points3D.ply &&
python train.py --config configs/dynerf/sear_steak.yaml --num_workers 6 --ply_path data/N3V/sear_steak/colmap_0/sparse/0/points3D.ply


sbatch --partition=suma_a6000 --gres=gpu:1 train_4dgs.sh bike1 &&
sbatch --partition=suma_a6000 --gres=gpu:1 train_4dgs.sh bike2 &&
sbatch --partition=suma_a6000 --gres=gpu:1 train_4dgs.sh corgi1 &&
sbatch --partition=suma_a6000 --gres=gpu:1 train_4dgs.sh corgi2 &&
sbatch --partition=suma_a6000 --gres=gpu:1 train_4dgs.sh dance1 &&
sbatch --partition=suma_a6000 --gres=gpu:1 train_4dgs.sh dance2
