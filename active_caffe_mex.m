function active_caffe_mex(gpu_id)
% active_caffe_mex(gpu_id, caffe_version)

    % set gpu in matlab
%   gpuDevice(gpu_id);%��Ϊmatlab��������GPU��������������caffe�л���caffe.set_device���ʴ˴������Ϊ�����á������ò����޴����֡�

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
    cd('D:\CAFFE\caffe\caffe-master\Build\x64\Release');%����mex64���ڵ�ַ
    caffe.set_device(gpu_id-1);
    %cd('D:\01MachineLearning\caffe\caffe-master\matlab\demo');%���ش����ļ���
end
