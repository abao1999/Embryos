%% include brightfield and fluorescence image into a single instance
%% output instances by the index of embryo
close all;clear;clc;
addpath('funcs');

bf_data_folder = 'video_bf_data'; % bright field data
fluo_data_folder = 'video_fluo_data'; % fluorescence data
listing = dir([bf_data_folder '\embryo*']);
addpath(bf_data_folder);
addpath(fluo_data_folder);

name = {listing.name};
str  = sprintf('%s#', name{:});
num  = sscanf(str, 'embryo_%d.mat#');
[dummy, index] = sort(num);
name = name(index);

%% create folders
instance_bf_gif = 'instance_gif_bf'; % create a folder to save instance bf gif
instance_fluo_gif = 'instance_gif_fluo'; % create a folder to save instance fluo gif
instance_data = 'instance_data_order'; % create a folder to save instance data
if ~exist(instance_bf_gif, 'dir')
    mkdir(instance_bf_gif);
end
if ~exist(instance_fluo_gif, 'dir')
    mkdir(instance_fluo_gif);
end
if ~exist(instance_data, 'dir')
    mkdir(instance_data);
end
addpath(instance_bf_gif);
addpath(instance_fluo_gif);
addpath(instance_data);

%%
instance_count = 1;
for embryo_idx = 1:length(listing)
    clear bf fluo data_bf data_fluo anno
    bf = load([bf_data_folder filesep listing(index(embryo_idx)).name]);
    fluo = load([fluo_data_folder filesep listing(index(embryo_idx)).name]);
    data_bf = bf.data;
    data_fluo = fluo.data;
    anno = bf.anno;
    
    t_num = size(data_bf,1);
    for t = 1:t_num
        bf_data = squeeze(data_bf(t,:,:,:));
        fluo_data = squeeze(data_fluo(t,:,:,:));
        label = anno(t);
        
        bf_gif_name = [instance_bf_gif filesep 'instance' num2str(instance_count) '_bf.gif'];
        fluo_gif_name = [instance_fluo_gif filesep 'instance' num2str(instance_count) '_fluo.gif'];
        mat_name = [instance_data filesep 'instance' num2str(instance_count) '.mat'];
        
        create_gif(bf_data,bf_gif_name); % create gif for each instance's bf image
        create_gif(fluo_data,fluo_gif_name); % create gif for each instance's fluo image
        save(mat_name,'bf_data','fluo_data','label'); % save mat for each instance
        
        instance_count = instance_count+1;
    end
end