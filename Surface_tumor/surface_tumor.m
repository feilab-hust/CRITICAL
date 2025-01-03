%% 该程序用于自动计算肺叶内肿瘤的分布情况。
%% 计算时取每个肿瘤内距离肺叶表面最近的点来计算Distance
%% 运行前需要先运行img_processing.m得到优化过的二值图lung_space_improve.tif，
%% 再保存splicing_Merge_v5.tif的绿色通道为tumor.tif
%% 还需要输入TumorSegment.ijm生成的csv统计表格
%% 程序会计算出每个肿瘤距离肺叶表面的距离，并记录在原统计表格的第一列中(会生成一个新的xlsx表格)

clc;clear
%% 设置参数（图像、csv表格均默认在bathpath路径下）
bathpath = 'D:\Data\sunlab_surface\20231227-wt-24m-3-lobe1-surface\1lobe-L_1';
imgname = 'lung_space_improve.tif';   % 肺叶分割二值图
imgpath = fullfile(bathpath,imgname);
tumorname = 'tumor.tif';   % 肿瘤分割二值图
tumorpath = fullfile(bathpath,tumorname);
pixel2um = 20.40;   % 1个pixel是(X20.64,Y20.64,Z20)um，取近似20.40um/pixel

%% 开始运行
disp('surface肿瘤分布程序开始运行');
t1 = tic;

% 获取图像信息
Info = imfinfo(imgpath);
Slice = size(Info,1);
Width = Info.Width;
Height = Info.Height;
 
% 读入图像
img = zeros(Height,Width,Slice);   % 声明一个空的三维的数组，用于读入图像信息
tumor = zeros(Height,Width,Slice);
for i = 1:Slice   % 逐帧地读入图像
    img(:,:,i) = imread(imgpath,i);
    tumor(:,:,i) = imread(tumorpath,i);
end
img = imcomplement(img);   % 注意：将二值图取反，需要肺内为0，肺外为1
[D,IDX] = bwdist(img);   % 得到距离肿瘤的距离矩阵D
disp(['该肺内的最大距离为：',num2str(max(D(:))*pixel2um),'um']);

% 读入.csv表格
csv_file = dir(fullfile(bathpath,'*.csv'));
csv_info = importdata(fullfile(bathpath,csv_file.name));   % 读取肺叶中肿瘤坐标的表格数据
csv_data = csv_info.data;   % 数据矩阵
csv_title = csv_info.textdata;   % 标题cell数组
csv_title = [{'距离(um)'}, csv_title];   % 第一列添加一列标题

% 循环遍历每一行数据
numRows = size(csv_data, 1);   % 行数
distance = zeros(numRows,1);   % 存放距离数据
for i = 1:numRows
    % 判断体积，太小的不考虑，距离记为-1
    % 暂时不剔除
%     Volume = csv_data(i,1);   % 得到第一列的体积信息
%     if Volume < 33510   % 体积阈值
%         distance(i) = -1;
%         continue;
%     end     
    % 得到后面3列坐标信息（XM,YM,ZM）
    xyz = csv_data(i,17:19);   
    % 得到肿瘤的中心坐标
    h = round(xyz(2))+1;   % YM   +1是为了让imageJ与matlab的索引对应
    w = round(xyz(1))+1;   % XM
    d = round(xyz(3))+1;   % ZM
    
    if tumor(h,w,d) == 0   % 若该肿瘤质心不在连通域内（通常主要是体积极小的肿瘤，偶尔有极个别是因为形状怪异）
        distance(i) = -1;
        continue;
    end
    
    n_space = tumor(h,w,d);   % 该肿瘤对应图像tumor中的第n_space个连通域
    % ind2sub：一维索引转n维坐标   sub2ind：n维坐标转一维索引
    %[h_list, w_list, d_list] = ind2sub(size(tumor), find(tumor == n_space));   % 得到上面区域内所有像素点在原图中对应的(h,w,d)
    %minDistance = min(D(sub2ind(size(D), h_list, w_list, d_list)));   % 得到该区域内所有像素点距离肺叶表面的最小距离
    minDistance = min(D(find(tumor == n_space)));
    % 将距离结果记录到第一列的列向量中
    distance(i) = minDistance * pixel2um;
end

csv_data = [distance,csv_data];   % 在第一列前插入判断结果

excel_savename = [csv_file.name((1:strlength(csv_file.name)-3)), 'xlsx'];
excel_savepath = fullfile(bathpath, excel_savename);
xlswrite(excel_savepath, csv_title, 'Sheet1', 'A1');
xlswrite(excel_savepath, csv_data, 'Sheet1', 'A2');


disp('程序运行结束，总用时为：');
toc(t1)
