clear;close all;
%% settings
folder = 'Test/Set2';
savepath = 'test.h5';
size_input = 33;
size_label = 33;
scale = 3;
stride = 33;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    if(size(image,3)==3)
        image = rgb2ycbcr(image);
    end
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    %im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');% change img from High-resolution to Low-resolution
    im_input = imnoise(im_label,'gaussian',0,0.038);% add Gaussian-noise 
%     poisson_scale = 3e10; % poisson noise
%     im_input = poisson_scale * imnoise(im_label/poisson_scale, 'poisson'); % poisson noise
    im_label = im_input - im_label;% generate mould

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);

            count=count+1;
            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_label;
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 2;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
