python main.py \
 -i ./test_imgs/figure7/input \
 -o ./outputs \
 --exp_name figure7 \
 --content_weight 8.0 --style_weight 1.7 --recon_weight 20 --face_model BFM --style_transfer_num 5 --add_noise --check_results_exist --save_inter_results

python main.py\
 -i ./test_imgs/figure8/input \
 -o ./outputs \
 -c ./test_imgs/figure8/ostec \
 --exp_name figure8 \
 --content_weight 8.0 --style_weight 1.7 --recon_weight 20 --face_model BFM --style_transfer_num 5 --add_noise --check_results_exist --save_inter_results