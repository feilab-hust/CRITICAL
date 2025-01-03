%% 该程序用于对肺的二值图进行进一步优化（对二值图先闭后开运算）
%% 输入的是肺叶分割的二值图lung_space.tif，通过ImageJ软件的Labkit插件获取。
%% 输出优化后的二值图lung_space_improve.tif


clc;clear
%% 设置参数（需设置二值图像的路径，csv表格默认在bathpath路径下）
bathpath = 'D:\Data\sunlab_surface\20231227-wt-24m-3-lobe1-surface\1lobe-L_1';
imgname = 'lung_space.tif';
savename = 'lung_space_improve.tif';
savepath = fullfile(bathpath,savename);
imgpath = fullfile(bathpath,imgname); %图像名称与路径

%% 开始运行
disp('lung_space图像优化程序开始运行');
t1 = tic;
% 获取图像信息
Info=imfinfo(imgpath);
Slice=size(Info,1);
Width=Info.Width;
Height=Info.Height;
 
% 读入图像
img=zeros(Height,Width,Slice);
for i=1:Slice
    img(:,:,i)=imread(imgpath,i);
end

% 膨胀腐蚀
disp('正在膨胀腐蚀');
tic
se1 = strel('sphere',7); % sphere是三维核，球形
se2 = strel('sphere',7); % sphere是三维核，球形
new_img = imopen(imclose(img,se1),se2);
toc

% 保存图像
disp('正在保存图像');
for i = 1:Slice   
    J=uint8(new_img(:,:,i)); 
    if i == 1     
        imwrite(J, savepath);   % 实测省略这行也行，不知道会不会有其他bug
    else
        imwrite(J, savepath,'WriteMode','append');   % 将多张图像写入一个多页tif文件
    end
end

disp('程序运行结束，总用时为：');
toc(t1)


