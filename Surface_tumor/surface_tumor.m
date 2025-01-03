%% �ó��������Զ������Ҷ�������ķֲ������
%% ����ʱȡÿ�������ھ����Ҷ��������ĵ�������Distance
%% ����ǰ��Ҫ������img_processing.m�õ��Ż����Ķ�ֵͼlung_space_improve.tif��
%% �ٱ���splicing_Merge_v5.tif����ɫͨ��Ϊtumor.tif
%% ����Ҫ����TumorSegment.ijm���ɵ�csvͳ�Ʊ��
%% ���������ÿ�����������Ҷ����ľ��룬����¼��ԭͳ�Ʊ��ĵ�һ����(������һ���µ�xlsx���)

clc;clear
%% ���ò�����ͼ��csv����Ĭ����bathpath·���£�
bathpath = 'D:\Data\sunlab_surface\20231227-wt-24m-3-lobe1-surface\1lobe-L_1';
imgname = 'lung_space_improve.tif';   % ��Ҷ�ָ��ֵͼ
imgpath = fullfile(bathpath,imgname);
tumorname = 'tumor.tif';   % �����ָ��ֵͼ
tumorpath = fullfile(bathpath,tumorname);
pixel2um = 20.40;   % 1��pixel��(X20.64,Y20.64,Z20)um��ȡ����20.40um/pixel

%% ��ʼ����
disp('surface�����ֲ�����ʼ����');
t1 = tic;

% ��ȡͼ����Ϣ
Info = imfinfo(imgpath);
Slice = size(Info,1);
Width = Info.Width;
Height = Info.Height;
 
% ����ͼ��
img = zeros(Height,Width,Slice);   % ����һ���յ���ά�����飬���ڶ���ͼ����Ϣ
tumor = zeros(Height,Width,Slice);
for i = 1:Slice   % ��֡�ض���ͼ��
    img(:,:,i) = imread(imgpath,i);
    tumor(:,:,i) = imread(tumorpath,i);
end
img = imcomplement(img);   % ע�⣺����ֵͼȡ������Ҫ����Ϊ0������Ϊ1
[D,IDX] = bwdist(img);   % �õ����������ľ������D
disp(['�÷��ڵ�������Ϊ��',num2str(max(D(:))*pixel2um),'um']);

% ����.csv���
csv_file = dir(fullfile(bathpath,'*.csv'));
csv_info = importdata(fullfile(bathpath,csv_file.name));   % ��ȡ��Ҷ����������ı������
csv_data = csv_info.data;   % ���ݾ���
csv_title = csv_info.textdata;   % ����cell����
csv_title = [{'����(um)'}, csv_title];   % ��һ�����һ�б���

% ѭ������ÿһ������
numRows = size(csv_data, 1);   % ����
distance = zeros(numRows,1);   % ��ž�������
for i = 1:numRows
    % �ж������̫С�Ĳ����ǣ������Ϊ-1
    % ��ʱ���޳�
%     Volume = csv_data(i,1);   % �õ���һ�е������Ϣ
%     if Volume < 33510   % �����ֵ
%         distance(i) = -1;
%         continue;
%     end     
    % �õ�����3��������Ϣ��XM,YM,ZM��
    xyz = csv_data(i,17:19);   
    % �õ���������������
    h = round(xyz(2))+1;   % YM   +1��Ϊ����imageJ��matlab��������Ӧ
    w = round(xyz(1))+1;   % XM
    d = round(xyz(3))+1;   % ZM
    
    if tumor(h,w,d) == 0   % �����������Ĳ�����ͨ���ڣ�ͨ����Ҫ�������С��������ż���м���������Ϊ��״���죩
        distance(i) = -1;
        continue;
    end
    
    n_space = tumor(h,w,d);   % ��������Ӧͼ��tumor�еĵ�n_space����ͨ��
    % ind2sub��һά����תnά����   sub2ind��nά����תһά����
    %[h_list, w_list, d_list] = ind2sub(size(tumor), find(tumor == n_space));   % �õ������������������ص���ԭͼ�ж�Ӧ��(h,w,d)
    %minDistance = min(D(sub2ind(size(D), h_list, w_list, d_list)));   % �õ����������������ص�����Ҷ�������С����
    minDistance = min(D(find(tumor == n_space)));
    % ����������¼����һ�е���������
    distance(i) = minDistance * pixel2um;
end

csv_data = [distance,csv_data];   % �ڵ�һ��ǰ�����жϽ��

excel_savename = [csv_file.name((1:strlength(csv_file.name)-3)), 'xlsx'];
excel_savepath = fullfile(bathpath, excel_savename);
xlswrite(excel_savepath, csv_title, 'Sheet1', 'A1');
xlswrite(excel_savepath, csv_data, 'Sheet1', 'A2');


disp('�������н���������ʱΪ��');
toc(t1)
