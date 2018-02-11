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
up_scale = 0;%�ü��߽�ߴ�
input_size = 256;%���붨��ߴ�

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
        %image = shave(image, [up_scale, up_scale]);%�и�����׼
        
        %% get noisemap & denoise Img
        lay_names=net.blob_names; %��ʾ������� 
        res=net.forward({im_input,image});
        data_map = feature_partvisual( net,1,1);
        label_map = feature_partvisual( net,2,1);

        feature_map = feature_partvisual( net,105,1);%8C output

        %im = data_map -feature_map;%ѵ������ͼʱʹ��
        %figure,imshow(im),title('-residual');%ѵ������ͼʱʹ��
        im = feature_map;%ѵ��ԭͼʱʹ��
        figure,imshow(im),title('MFCNN');%ѵ������ͼʱʹ��

        %% compute PSNR
        image = uint8(image * 255); %��Ӧԭʼͼ�� ת��Ϊ0-255ͼ�����
        im = uint8(im * 255);%�����ͼ��
        data_map = uint8(data_map * 255);%�����������ݲ�ͼ��,������ͼ��
        psnr_noise_poisson=compute_psnr(image,data_map);
        psnr_denoise_poisson=compute_psnr(image,im);

        fprintf('img_name: %s\n', filepaths(i).name);
        fprintf('add noise : %d * e10\n', poisson_parameter);
        fprintf('PSNR for add noise: %f dB\n', psnr_noise_poisson);
        fprintf('PSNR for denoise: %f dB\n', psnr_denoise_poisson);
             
    end
end


    


