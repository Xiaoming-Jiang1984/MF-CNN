function active_caffe_mex(gpu_id)
% active_caffe_mex(gpu_id, caffe_version)

    % set gpu in matlab
%   gpuDevice(gpu_id);%此为matlab设置运算GPU器件函数，因在caffe中还有caffe.set_device，故此处不清楚为何设置。不设置测试无错误字。

%     if ~exist('caffe_version', 'var') || isempty(caffe_version)
%         caffe_version = 'caffe';
%     end
%     cur_dir = pwd;
%     caffe_dir = fullfile(pwd, 'external', 'caffe', 'matlab', caffe_version);
%     
%     if ~exist(caffe_dir, 'dir')
%         warning('Specified caffe folder (%s) is not exist, change to default one (%s)', ...
%             caffe_dir, fullfile(pwd, 'external', 'caffe', 'matlab'));
%         caffe_dir = fullfile(pwd, 'external', 'caffe', 'matlab');
%     end
    
%     addpath(genpath(caffe_dir));
    cd('D:\CAFFE\caffe\caffe-master\Build\x64\Release');%进入mex64所在地址
    caffe.set_device(gpu_id-1);
    %cd('D:\01MachineLearning\caffe\caffe-master\matlab\demo');%返回代码文件夹
end
