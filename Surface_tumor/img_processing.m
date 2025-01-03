%% �ó������ڶԷεĶ�ֵͼ���н�һ���Ż����Զ�ֵͼ�ȱպ����㣩
%% ������Ƿ�Ҷ�ָ�Ķ�ֵͼlung_space.tif��ͨ��ImageJ�����Labkit�����ȡ��
%% ����Ż���Ķ�ֵͼlung_space_improve.tif


clc;clear
%% ���ò����������ö�ֵͼ���·����csv���Ĭ����bathpath·���£�
bathpath = 'D:\Data\sunlab_surface\20231227-wt-24m-3-lobe1-surface\1lobe-L_1';
imgname = 'lung_space.tif';
savename = 'lung_space_improve.tif';
savepath = fullfile(bathpath,savename);
imgpath = fullfile(bathpath,imgname); %ͼ��������·��

%% ��ʼ����
disp('lung_spaceͼ���Ż�����ʼ����');
t1 = tic;
% ��ȡͼ����Ϣ
Info=imfinfo(imgpath);
Slice=size(Info,1);
Width=Info.Width;
Height=Info.Height;
 
% ����ͼ��
img=zeros(Height,Width,Slice);
for i=1:Slice
    img(:,:,i)=imread(imgpath,i);
end

% ���͸�ʴ
disp('�������͸�ʴ');
tic
se1 = strel('sphere',7); % sphere����ά�ˣ�����
se2 = strel('sphere',7); % sphere����ά�ˣ�����
new_img = imopen(imclose(img,se1),se2);
toc

% ����ͼ��
disp('���ڱ���ͼ��');
for i = 1:Slice   
    J=uint8(new_img(:,:,i)); 
    if i == 1     
        imwrite(J, savepath);   % ʵ��ʡ������Ҳ�У���֪���᲻��������bug
    else
        imwrite(J, savepath,'WriteMode','append');   % ������ͼ��д��һ����ҳtif�ļ�
    end
end

disp('�������н���������ʱΪ��');
toc(t1)


