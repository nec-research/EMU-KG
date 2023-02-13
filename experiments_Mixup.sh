# FB15k-237
# uniform sampling
# DM
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/FB15k-237 --model DistMult -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/DM_KB237_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/FB15k-237 --model DistMult -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/DM_KB237_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# TransE
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/FB15k-237 --model TransE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/TE_KB237_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/FB15k-237 --model TransE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/TE_KB237_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# ComplEx
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/FB15k-237 --model ComplEx -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/CE_KB237_uni_Mixup_0 -if_CE 1 -de -dr --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/FB15k-237 --model ComplEx -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/CE_KB237_uni_Mixup_0 -if_CE 1 -de -dr --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# RotatE
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/FB15k-237 --model RotatE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/RE_KB237_uni_Mixup_0 -if_CE 1 -de --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/FB15k-237 --model RotatE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/RE_KB237_uni_Mixup_0 -if_CE 1 -de --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# WN18
# uniform sampling
# DM
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/wn18 --model DistMult -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/DM_WN18_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/wn18 --model DistMult -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/DM_WN18_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# TransE
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/wn18 --model TransE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/TE_WN18_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/wn18 --model TransE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/TE_WN18_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# ComplEx
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/wn18 --model ComplEx -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/CE_WN18_uni_Mixup_0 -if_CE 1 -de -dr --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/wn18 --model ComplEx -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/CE_WN18_uni_Mixup_0 -if_CE 1 -de -dr --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# RotatE
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/wn18 --model RotatE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/RE_WN18_uni_Mixup_0 -if_CE 1 -de --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/wn18 --model RotatE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/RE_WN18_uni_Mixup_0 -if_CE 1 -de --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# WN18rr
# uniform sampling
# DM
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/wn18rr --model DistMult -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/DM_WN18rr_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/wn18rr --model DistMult -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/DM_WN18rr_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# TransE
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/wn18rr --model TransE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/TE_WN18rr_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/wn18rr --model TransE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/TE_WN18rr_uni_Mixup_0 -if_CE 1 --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# ComplEx
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/wn18rr --model ComplEx -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/CE_WN18rr_uni_Mixup_0 -if_CE 1 -de -dr --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/wn18rr --model ComplEx -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 1.e-1 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/CE_WN18rr_uni_Mixup_0 -if_CE 1 -de -dr --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
# RotatE
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_train --do_valid \
--data_path data/wn18rr --model RotatE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/RE_WN18rr_uni_Mixup_0 -if_CE 1 -de --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
CUDA_VISIBLE_DEVICES=1 python3 -u codes/run.py --cuda --do_test \
--data_path data/wn18rr --model RotatE -n 256 -b 1000 -d 100 -g 24.0 -a 1. \
-lr 5.e-2 --max_steps 100000 -khop 3 -nrw 1000 \
-save models/RE_WN18rr_uni_Mixup_0 -if_CE 1 -de --valid_steps 500 --log_steps 250 \
-if_Mutup 0.39 -neg_label 0.53 -CE_coef 0.11 -if_Mixup 1.
#-if_Mutup 0.3 -neg_label 0.55 -CE_coef 0.25
