%% 该代码用于自动从.csv表格中读取肿瘤坐标信息并crop出来
clc;clear;
t0 = tic;
disp('--------------------crop程序开始运行--------------------');

%% 设置初始参数
s1 = 4; % scquence2stack时的下采样倍数
% s2 = 2; % 计数时的下采样倍数，1.26倍的数据没用到
Dir1='I:\data4';   % 第一级文件夹目录
SAVE_Dir1='J:\savedata4';
Volume_thr = 33510;   % 体积阈值(um3)，不需要时设为0即可

%% 遍历路径
temp1 = dir(Dir1);   % 得到第一级目录下的所有文件
temp1 = temp1(3:size(temp1,1));   % ！！！注意:前两项为.和..，代表该级和上一级目录，需要删掉
View_name1 = {temp1.name};   % 文件名（每组需要处理的数据）
view_num1 = size(View_name1, 2);    % 数据组数
for n1 = 1:view_num1   % 处理每组数据
    % 第二级目录Dir2
    Dir2 = fullfile(Dir1,View_name1{n1});   % 得到第二级目录，View_name1{n1}:41_1-lacz_16W(1.26x)-20230322
    SAVE_Dir2 = fullfile(SAVE_Dir1,View_name1{n1});
    temp2 = dir(Dir2);   % 得到第二级目录下的所有文件
    temp2 = temp2(3:size(temp2,1));
    View_name2 = {temp2.name};   % 文件名（每个肺叶）
    view_num2 = size(View_name2, 2);    % 肺叶个数
    for n2 = 1:view_num2   % 处理每个肺叶
        % 第三级目录Dir3：有一个excel表格，在此处读取表格数据。
        Dir3 = fullfile(Dir2,View_name2{n2});   % 得到第三级目录↓（即excel文件所在的文件夹目录） 
        SAVE_Dir3 = fullfile(SAVE_Dir2,View_name2{n2});   % V_n{n2}:41_1-lacz_16W-488-40ms-5um-1.26x-lobe1-L_1
        % 得到保存crop肿瘤的名称
        temp_name = split(View_name2{n2},'-');
        name_lung = cell2mat(temp_name(1)); % 41_1
        name_gene = cell2mat(temp_name(2)); % lacz_16W
        name_lobe = cell2mat(temp_name(length(temp_name)-1)); % lobe1
        name_LR = cell2mat(temp_name(length(temp_name))); % L_1
        tumor_name = [name_lung,'-',name_gene,'-',name_lobe,'-',name_LR,'-']; % 保存的crop肿瘤名称
        % 找到路径下的excel表格
        excel_file = dir(fullfile(SAVE_Dir3,'*.csv'));
        xyz = table2array(readtable(fullfile(SAVE_Dir3,excel_file.name)));   % 读取肺叶中肿瘤坐标的excel表格数据！！！ 
        
        % 根据文件夹名字判断左右光路
        LR = name_LR;   % 'L_1' or 'left_1   right_1'
        LR = LR(1);   % 'L'or'l'or'R'or'r'

        %% 处理未拼接的1.26倍图像（仅在一张原图上crop，不用考虑黑边以及肿瘤跨图问题）         
        Dir4 = fullfile(Dir3,'Default');   % 得到最后一级Default目录
        temp4 = dir(fullfile(Dir4,'*.tif'));   % 得到Default文件夹里所有存成scquence的tif 
        View_name4 = {temp4.name};   % 得到所有tif图像名       
        % 读取图像
        tic
        disp(['----------正在读入图像:“',View_name2{n2},'”----------']);
        Slice = size(View_name4, 2);    % 图像个数（即depth）           
        for i = 1:Slice         
            filepath = fullfile(Dir4,View_name4{i});   % 第i层图像的路径   
            if i==1
                Info = imfinfo(filepath);
                Width = Info.Width;     % x（列数）
                Height = Info.Height;   % y（行数）
                img = zeros(Height,Width,Slice);   % 声明一个空的三维的数组，用于读入图像信息。
            end                             
            img(:,:,i) = imread(filepath);   % 一层一层地读入图像，注意这里读的是scquence                    
        end
        toc
        % 对每个肿瘤进行crop操作
        for j = 1:size(xyz,1)   % 统计肿瘤个数
            tic
            block_pixel = xyz(j,20:25); % 得到第j个肿瘤的体积坐标（1*6），依次为起点x,y,z,width,height,depth

            volume = xyz(j,1);   % 不crop体积小于阈值的肿瘤
            if volume < Volume_thr
                continue;
            end

            w_min=block_pixel(1)*s1;  w_max=(block_pixel(1)+block_pixel(4)-1)*s1;
            h_min=block_pixel(2)*s1;  h_max=(block_pixel(2)+block_pixel(5)-1)*s1;
            d_min=block_pixel(3)*s1;  d_max=(block_pixel(3)+block_pixel(6)-1)*s1;
            % ↓增强鲁棒性：防止坐标转换时有个别坐标略微超出原图范围
            w_max = min(w_max,Width);   w_min = max(w_min,1);
            h_max = min(h_max,Height);  h_min = max(h_min,1);
            d_max = min(d_max,Slice);   d_min = max(d_min,1);             
            
            % 判断是L还是R光路
            if (LR == 'L' || LR == 'l') && (h_max+h_min) >= Height   % 若是左光路，则图像下半部分的不处理
                continue;
            elseif (LR == 'R' || LR == 'r') && (h_max+h_min) < Height   % 若是右光路，则图像上半部分的不处理
                continue;
            end
            
            disp(['正在crop第',num2str(j,'%d'),'个肿瘤']);
            Image_crop = img(h_min:h_max,w_min:w_max,d_min:d_max);   % (h,w,d)

            % 保存每个crop的肿瘤图像
            save_path = fullfile(SAVE_Dir3,'save_tumor');   % 存在SAVE下
            if ~exist(save_path,'dir')
                mkdir(save_path); % 生成文件夹，这张图像crop的肿瘤装一个文件夹内
            end                
            for i = 1:(d_max-d_min+1)   % crop的depth
                J=uint16(Image_crop(:,:,i));            % 一层一层地保存图像 
                imwrite(J,fullfile(save_path,[tumor_name,num2str(j,'%04d'),'.tif']),'WriteMode','Append'); %存为stack
            end
            toc
        end                                                                             
    end 
end
disp('--------------------crop程序运行结束--------------------');
disp('--------------------程序总用时如下：--------------------');
toc(t0)