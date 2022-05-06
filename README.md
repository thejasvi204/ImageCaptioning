# SGAE/ pytorch 0.4.0
Auto-Encoding Scene Graphs for Image Captioning, CVPR 2019

# Acknowledgement
This repository is derived from work of Yang Xu et al. at https://github.com/yangxuntu/SGAE

This code is implemented based on Ruotian Luo's implementation of self critical sequence training in https://github.com/ruotianluo/self-critical.pytorch.

And we use the visual features provided by paper Bottom-up and top-down attention for image captioning and visual question answering in https://github.com/peteanderson80/bottom-up-attention.


# Training the model
1.After downloading the codes and meta data, you can train the model by using the following code:
```
python train_mem.py --id id66 --caption_model lstm_mem4 --input_json data/cocobu2.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_rela_dir data/cocobu_sg_img --input_ssg_dir data/coco_spice_sg2 --input_label_h5 data/cocobu2_label.h5 --sg_dict_path data/spice_sg_dict2.npz --batch_size 50 --accumulate_number 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 5 --scheduled_sampling_start 0 --checkpoint_path id66 --save_checkpoint_every 5000 --val_images_use 5000 --max_epochs 150 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 40 --train_split train --memory_size 10000 --memory_index c --step2_train_after 10 --step3_train_after 20 --use_rela 0 --gpu 5
```
Important notes: I reorganized and optimized the code recently and found that even without MGCN, this code can achieve 127.8 CIDEr-D score. But you need to have a 16G gpu like DGX. If your memory is not enough, you should change --batch_size from 50 to 25, and --accumulate_number from 2 to 4, which can make the batch size be 100. But I found that these two different settings will lead to different performances.

2.You can also go to my google drive to download two well-trained models which are modelid740072 and modelid640075 for getting about 128.3 CIDEr-D scores, and these two models are trained by using the above scripts.

3.The details of parameters:

--id: the id of your model, which is usually set as the same as check point, which is helpful for you to train from the check point.

--batch_size, --accumulate_number: these two parameters are set for users who do not have large gpu, if you want to set batch size as 100, you can set batch_size as 50, and set accumulate_number as 2, also you can set batch_size as 20 and accumulate_number as 5. Importantly, they are not totally equal to set batch_size as 100 and accumulate_number as 1, the bigger the bathc_size is, the higher the performance.

--self_critical_after: when reinforcement leanring begins, if this value is set as 40, it means that after training 40 epoches, the reinforcement loss is used. Generally, if you want to have a good CIDEr-D score, you should use cross entropy loss first and then use reinforcement loss.

--step2_train_after: when the dictionary is learned, for example, if this value is set as 10, then before 10 epochs, only the decoder is trained by sentence scene graphs and the dictionary is not learned. 

--step3_train_after: when image captioning encoder-decoder is learned, for example, if this value is set as 20, then before 20 epochs, only sentence scene graphs are used to learn the dictionary, and after 20 epochs, the sentence scene graphs are no longer used and the image encoder-decoder is trained.

--use_rela: whether use image scene graph

4.Tranining from checkpoints.
The codes provide the ability of training from checkpoints. For example, if you want to train the model from one checkpoint, say, 22, you can use the following code to continute:
```
python train_mem.py --id id66 --caption_model lstm_mem4 --input_json data/cocobu2.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_rela_dir data/cocobu_sg_img --input_ssg_dir data/coco_spice_sg2 --input_label_h5 data/cocobu2_label.h5 --sg_dict_path data/spice_sg_dict2.npz --batch_size 50 --accumulate_number 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 5 --scheduled_sampling_start 0 --checkpoint_path id66 --save_checkpoint_every 5000 --val_images_use 5000 --max_epochs 150 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 40 --train_split train --memory_size 10000 --memory_index c --step2_train_after 10 --step3_train_after 20 --use_rela 0 --gpu 5 --start_from 22 --memory_cell_path id66/memory_cellid660022.npz
```
the parameter start_from and memory_cell_path are used for training from checkpoints.

# Evaluating the model
1.After training the model or downloading the well-trained model, you can evaluate them by using the following code:
```
python eval_mem.py --dump_images 0 --num_images 5000 --model id66/modelid660066.pth --infos_path id66/infos_id660066.pkl --language_eval 1 --beam_size 5 --split test --index_eval 1 --use_rela 0 --training_mode 2 --memory_cell_path id66/memory_cellid660066.npz --sg_dict_path data/spice_sg_dict2.npz --input_ssg_dir data/coco_spice_sg2 --batch_size 50
```
what you need to do is to switch the model id with your id, like 66 to 01, and change the number like 0066 to your trained model, like 0066 to 0001.

# Generating Scene graphs:
1.For sentence scene graph, you can directly download the revised code in spice-1.0.jar and create_coco_sg.py, put spice-1.0.jar in /coco-caption/pycocoevalcap/spice, then you should set coco_use as coco_train or coco_val in file create_coco_sg.py, then run this code and the sentence scene graphs are generated in /coco-caption/pycocoevalcap/spice/sg.json.

2.Then use process_spice_sg.py to process sg.json.

3.For image scene graph, you can directly download the code provided by https://github.com/rowanz/neural-motifs for generating image scene graphs.
