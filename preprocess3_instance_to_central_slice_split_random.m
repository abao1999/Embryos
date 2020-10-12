%% output 2D central slice images
% split train/test after randomizing the embryo index
% bf and fluo images are both scaled to [0,255]
close all;clear;clc;

T = readtable('video_data_time_info.xlsx');
t_num = T.t_num;

rng(13);
p = randperm(length(t_num));
t_num_random = t_num(p);

instance_cum_random = cumsum(t_num_random);
split_point = instance_cum_random(end)*0.8;
temp = abs(instance_cum_random-split_point);
idx = find(temp==min(temp));

% normal t list grouped by embryo
instance_cum = cumsum(t_num);
instance_cum = [0 instance_cum'];
t_list = {};
for k=1:(length(instance_cum)-1)
    t_list{k} = [(instance_cum(k)+1):instance_cum(k+1)];
end
% randomized t list grouped by embryo 
train_list_random = [];
for k = p(1:idx)
    train_list_random = [train_list_random t_list{k}];
end
test_list_random = [];
for k = p((idx+1):end)
    test_list_random = [test_list_random t_list{k}];
end

%% specify the 3D instance folder 
instance_data = 'instance_data_order'; % folder to save instance data
addpath(instance_data);

%% save 2D central slice images
bf_folder_name = 'embryo-58-bf-image-1503-order-random';
fluo_folder_name = 'embryo-58-fluo-image-1503-order-random';

mkdir(bf_folder_name);
mkdir(fluo_folder_name);
mkdir([bf_folder_name filesep 'train']);
mkdir([bf_folder_name filesep 'test']);
mkdir([fluo_folder_name filesep 'train']);
mkdir([fluo_folder_name filesep 'test']);
mkdir([bf_folder_name filesep 'train' filesep 'no']);
mkdir([bf_folder_name filesep 'train' filesep 'yes']);
mkdir([bf_folder_name filesep 'test' filesep 'no']);
mkdir([bf_folder_name filesep 'test' filesep 'yes']);
mkdir([fluo_folder_name filesep 'train' filesep 'no']);
mkdir([fluo_folder_name filesep 'train' filesep 'yes']);
mkdir([fluo_folder_name filesep 'test' filesep 'no']);
mkdir([fluo_folder_name filesep 'test' filesep 'yes']);

%% bf and fluo images are both scaled from [img.min, img.max] to [0,255]
central_slice_idx = 8;
for k = 1:length(train_list_random)
    load([instance_data filesep 'instance' num2str(train_list_random(k)) '.mat']);
    bf_img = bf_data(:,:,central_slice_idx);
    fluo_img = fluo_data(:,:,central_slice_idx);
    
    last_name = ['instance' num2str(train_list_random(k)) '.png'];
    if label == 0
        bf_img_name = [bf_folder_name filesep 'train' filesep 'no' filesep last_name];
        fluo_img_name = [fluo_folder_name filesep 'train' filesep 'no' filesep last_name];
    else
        bf_img_name = [bf_folder_name filesep 'train' filesep 'yes' filesep last_name];
        fluo_img_name = [fluo_folder_name filesep 'train' filesep 'yes' filesep last_name];
    end
    % output image
    imwrite(uint8(255*mat2gray(bf_img)),bf_img_name);
    imwrite(uint8(255*mat2gray(fluo_img)),fluo_img_name);
end

for k = 1:length(test_list_random)
    load([instance_data filesep 'instance' num2str(test_list_random(k)) '.mat']);
    bf_img = bf_data(:,:,central_slice_idx);
    fluo_img = fluo_data(:,:,central_slice_idx);
    
    last_name = ['instance' num2str(test_list_random(k)) '.png'];
    if label == 0
        bf_img_name = [bf_folder_name filesep 'test' filesep 'no' filesep last_name];
        fluo_img_name = [fluo_folder_name filesep 'test' filesep 'no' filesep last_name];
    else
        bf_img_name = [bf_folder_name filesep 'test' filesep 'yes' filesep last_name];
        fluo_img_name = [fluo_folder_name filesep 'test' filesep 'yes' filesep last_name];
    end
    % output image
    imwrite(uint8(255*mat2gray(bf_img)),bf_img_name);
    imwrite(uint8(255*mat2gray(fluo_img)),fluo_img_name);
end

%% print out the data summary
fprintf('Train: %d embryo videos, %d images\n', idx, length(train_list_random));
fprintf('Test: %d embryo videos, %d images\n', length(p)-idx, length(test_list_random));
T_random = T(p,:);
writetable(T_random,[bf_folder_name filesep 'embryo_order.xlsx']);