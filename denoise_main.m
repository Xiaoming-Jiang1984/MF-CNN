clc;clear; close all;
%% active caffe and path(w.S)
sysPath = pwd  % show current project's path
gpu_id = 1; %set gpu
active_caffe_mex(gpu_id);  %active caffe mex
caffe.reset_all();
cd(sysPath);
%% nets settings
folder = sysPath;
model = [folder '\MFCNN_deploy.prototxt'];
weights = [folder '\*.caffemodel'];% caffe model
%load model using deploy.prototxt
net = caffe.Net(model,weights,'test');


%% load img & process
folder_test = 'Test';
filepaths = dir(fullfile(folder_test,'*.png'));
up_scale = 0;%裁剪边界尺寸
input_size = 256;%输入定义尺寸

for i = 1 : length(filepaths)
    
    originImg = imread(fullfile(folder_test,filepaths(i).name)); 

	if(size(originImg,1)~= input_size || size(originImg,2)~= input_size)  
        originImg = imresize(originImg , [input_size,input_size]);
    end

    if(size(originImg,3) ==3) 
        originImg = rgb2ycbcr(originImg);
        originImg = im2double(originImg(:, :, 1));
    else
        originImg = im2double(originImg(:, :, 1));
    end
   
    for poisson_parameter = 3:3
        image = originImg;
        im_input = image;
        poisson_scale = 1e10 * poisson_parameter; % poisson noise
        im_input = poisson_scale * imnoise(image/poisson_scale, 'poisson'); % poisson noise       
        %im_input = imnoise(image,'gaussian',0,0.038);% add Gaussian-noise
        %image = shave(image, [up_scale, up_scale]);%切割到输出标准
        
        %% get noisemap & denoise Img
        lay_names=net.blob_names; %显示网络层名 
        res=net.forward({im_input,image});
        data_map = feature_partvisual( net,1,1);
        label_map = feature_partvisual( net,2,1);

        feature_map = feature_partvisual( net,105,1);%8C output

        %im = data_map -feature_map;%训练残余图时使用
        %figure,imshow(im),title('-residual');%训练残余图时使用
        im = feature_map;%训练原图时使用
        figure,imshow(im),title('MFCNN');%训练残余图时使用

        %% compute PSNR
        image = uint8(image * 255); %对应原始图像 转换为0-255图像矩阵
        im = uint8(im * 255);%处理后图像
        data_map = uint8(data_map * 255);%网络输入数据层图像,加噪后的图像
        psnr_noise_poisson=compute_psnr(image,data_map);
        psnr_denoise_poisson=compute_psnr(image,im);

        fprintf('img_name: %s\n', filepaths(i).name);
        fprintf('add noise : %d * e10\n', poisson_parameter);
        fprintf('PSNR for add noise: %f dB\n', psnr_noise_poisson);
        fprintf('PSNR for denoise: %f dB\n', psnr_denoise_poisson);
             
    end
end


    


