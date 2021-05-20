# Deep Hough Transform

Code accompanying the paper "Deep Hough Transform for Semantic Line Detection" (ECCV 2020, PAMI 2021).
[arXiv2003.04676](https://arxiv.org/abs/2003.04676) | [Online Demo](http://mc.nankai.edu.cn/dht) | [Project page](http://mmcheng.net/dhtline) | [New dataset](http://kaizhao.net/nkl) | [Line Annotator](https://github.com/Hanqer/lines-manual-labeling)



To install deep-hough, run the following commands.
```sh
cd deep-hough-transform
cd model/_cdht
python setup.py build 
python setup.py install --user
```

Then run python script to generate parametric space label.
```sh
cd deep-hough-tranfrom
python data/prepare_data_JTLEE.py --root './data/ICCV2017_JTLEE_images/' --label './data/ICCV2017_JTLEE_gtlines_all' --save-dir './data/training/JTLEE_resize_100_100/' --list './data/training/JTLEE.lst' --prefix 'JTLEE_resize_100_100' --fixsize 400 --numangle 100 --numrho 100
```

### Training
Following the default config file 'config.yml', you can arbitrarily modify hyperparameters.
Then, run the following command.
```sh
python train.py
```

### Forward
Generate visualization results and save coordinates to _.npy file.
```sh
CUDA_VISIBLE_DEVICES=0 python forward.py --model （your_best_model.pth） --tmp (your_result_save_dir)
```

### Evaluate
Test the EA-score on SEL dataset. After forwarding the model and get the coordinates files. Run the following command to produce EA-score.
```sh
python test.py --pred result/debug/visualize_test/(change to your onw path which includes _.npy files) --gt gt_path/include_txt
```

### License
Our source code is free for non-commercial usage. Please contact us if you want to use it for comercial usage.

