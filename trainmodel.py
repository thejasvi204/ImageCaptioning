python train_mem.py --id id01 --caption_model lstm_mem4 --input_json data/cocobu2.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_rela_dir data/cocobu_sg_img --input_ssg_dir data/coco_spice_sg2 --input_label_h5 data/cocobu2_label.h5 --sg_dict_path data/spice_sg_dict2.npz --batch_size 50 --accumulate_number 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 5 --scheduled_sampling_start 0 --checkpoint_path id01 --save_checkpoint_every 5000 --val_images_use 500 --max_epochs 150 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 40 --train_split train --memory_size 10000 --memory_index c --step2_train_after 10 --step3_train_after 20 --use_rela 0 --gpu 5 --memory_cell_path id01/memory_cellid010022.npz