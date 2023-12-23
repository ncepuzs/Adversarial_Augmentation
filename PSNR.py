import os
# --------
# baseline
# --------
# os.system("python train_inversion_PSNR.py --norm 'noadv' --penalty 'no'  --epochs 500 \
#     --victim CNN4 --shadow ResNet \
#     --path_out Inversion_Models/PSNR/CNN/baseline/ \
#     --nz 100 --truncation 100 --lr 0.002 --early_stop 100 --adv_param '(0,0)' --lambda_pen 0\
#     --loss_mode 'PSNR'")

os.system("python Re_identification.py --norm 'noadv' --penalty 'no' \
    --path_out Inversion_Models/PSNR/CNN/baseline/ReId/ --path_Inv Inversion_Models/PSNR/CNN/baseline/ \
    --nz 100 --adv_param '(0,0)' --lambda_pen 0")

# --------
# our
# --------
# os.system("python train_inversion_PSNR.py --norm '2' --penalty 'yes'  --epochs 500 \
#     --victim CNN4 --shadow ResNet --train_or_not train\
#     --path_out Inversion_Models/PSNR/CNN/our/ \
#     --nz 100 --truncation 100 --lr 0.002 --early_stop 100 --adv_param '(300,0.02)' --lambda_pen 0.005\
#     --loss_mode 'PSNR'")

os.system("python Re_identification.py --norm '2' --penalty 'yes' \
    --path_out Inversion_Models/PSNR/CNN/our/ReId/ --path_Inv Inversion_Models/PSNR/CNN/our/ \
    --nz 100 --adv_param '(300,0.1)' --lambda_pen 0.005")

