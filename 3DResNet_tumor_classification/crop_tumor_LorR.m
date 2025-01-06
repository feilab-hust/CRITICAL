%% �ô��������Զ���.csv����ж�ȡ����������Ϣ��crop����
clc;clear;
t0 = tic;
disp('--------------------crop����ʼ����--------------------');

%% ���ó�ʼ����
s1 = 4; % scquence2stackʱ���²�������
% s2 = 2; % ����ʱ���²���������1.26��������û�õ�
Dir1='I:\data4';   % ��һ���ļ���Ŀ¼
SAVE_Dir1='J:\savedata4';
Volume_thr = 33510;   % �����ֵ(um3)������Ҫʱ��Ϊ0����

%% ����·��
temp1 = dir(Dir1);   % �õ���һ��Ŀ¼�µ������ļ�
temp1 = temp1(3:size(temp1,1));   % ������ע��:ǰ����Ϊ.��..������ü�����һ��Ŀ¼����Ҫɾ��
View_name1 = {temp1.name};   % �ļ�����ÿ����Ҫ��������ݣ�
view_num1 = size(View_name1, 2);    % ��������
for n1 = 1:view_num1   % ����ÿ������
    % �ڶ���Ŀ¼Dir2
    Dir2 = fullfile(Dir1,View_name1{n1});   % �õ��ڶ���Ŀ¼��View_name1{n1}:41_1-lacz_16W(1.26x)-20230322
    SAVE_Dir2 = fullfile(SAVE_Dir1,View_name1{n1});
    temp2 = dir(Dir2);   % �õ��ڶ���Ŀ¼�µ������ļ�
    temp2 = temp2(3:size(temp2,1));
    View_name2 = {temp2.name};   % �ļ�����ÿ����Ҷ��
    view_num2 = size(View_name2, 2);    % ��Ҷ����
    for n2 = 1:view_num2   % ����ÿ����Ҷ
        % ������Ŀ¼Dir3����һ��excel����ڴ˴���ȡ������ݡ�
        Dir3 = fullfile(Dir2,View_name2{n2});   % �õ�������Ŀ¼������excel�ļ����ڵ��ļ���Ŀ¼�� 
        SAVE_Dir3 = fullfile(SAVE_Dir2,View_name2{n2});   % V_n{n2}:41_1-lacz_16W-488-40ms-5um-1.26x-lobe1-L_1
        % �õ�����crop����������
        temp_name = split(View_name2{n2},'-');
        name_lung = cell2mat(temp_name(1)); % 41_1
        name_gene = cell2mat(temp_name(2)); % lacz_16W
        name_lobe = cell2mat(temp_name(length(temp_name)-1)); % lobe1
        name_LR = cell2mat(temp_name(length(temp_name))); % L_1
        tumor_name = [name_lung,'-',name_gene,'-',name_lobe,'-',name_LR,'-']; % �����crop��������
        % �ҵ�·���µ�excel���
        excel_file = dir(fullfile(SAVE_Dir3,'*.csv'));
        xyz = table2array(readtable(fullfile(SAVE_Dir3,excel_file.name)));   % ��ȡ��Ҷ�����������excel������ݣ����� 
        
        % �����ļ��������ж����ҹ�·
        LR = name_LR;   % 'L_1' or 'left_1   right_1'
        LR = LR(1);   % 'L'or'l'or'R'or'r'

        %% ����δƴ�ӵ�1.26��ͼ�񣨽���һ��ԭͼ��crop�����ÿ��Ǻڱ��Լ�������ͼ���⣩         
        Dir4 = fullfile(Dir3,'Default');   % �õ����һ��DefaultĿ¼
        temp4 = dir(fullfile(Dir4,'*.tif'));   % �õ�Default�ļ��������д��scquence��tif 
        View_name4 = {temp4.name};   % �õ�����tifͼ����       
        % ��ȡͼ��
        tic
        disp(['----------���ڶ���ͼ��:��',View_name2{n2},'��----------']);
        Slice = size(View_name4, 2);    % ͼ���������depth��           
        for i = 1:Slice         
            filepath = fullfile(Dir4,View_name4{i});   % ��i��ͼ���·��   
            if i==1
                Info = imfinfo(filepath);
                Width = Info.Width;     % x��������
                Height = Info.Height;   % y��������
                img = zeros(Height,Width,Slice);   % ����һ���յ���ά�����飬���ڶ���ͼ����Ϣ��
            end                             
            img(:,:,i) = imread(filepath);   % һ��һ��ض���ͼ��ע�����������scquence                    
        end
        toc
        % ��ÿ����������crop����
        for j = 1:size(xyz,1)   % ͳ����������
            tic
            block_pixel = xyz(j,20:25); % �õ���j��������������꣨1*6��������Ϊ���x,y,z,width,height,depth

            volume = xyz(j,1);   % ��crop���С����ֵ������
            if volume < Volume_thr
                continue;
            end

            w_min=block_pixel(1)*s1;  w_max=(block_pixel(1)+block_pixel(4)-1)*s1;
            h_min=block_pixel(2)*s1;  h_max=(block_pixel(2)+block_pixel(5)-1)*s1;
            d_min=block_pixel(3)*s1;  d_max=(block_pixel(3)+block_pixel(6)-1)*s1;
            % ����ǿ³���ԣ���ֹ����ת��ʱ�и���������΢����ԭͼ��Χ
            w_max = min(w_max,Width);   w_min = max(w_min,1);
            h_max = min(h_max,Height);  h_min = max(h_min,1);
            d_max = min(d_max,Slice);   d_min = max(d_min,1);             
            
            % �ж���L����R��·
            if (LR == 'L' || LR == 'l') && (h_max+h_min) >= Height   % �������·����ͼ���°벿�ֵĲ�����
                continue;
            elseif (LR == 'R' || LR == 'r') && (h_max+h_min) < Height   % �����ҹ�·����ͼ���ϰ벿�ֵĲ�����
                continue;
            end
            
            disp(['����crop��',num2str(j,'%d'),'������']);
            Image_crop = img(h_min:h_max,w_min:w_max,d_min:d_max);   % (h,w,d)

            % ����ÿ��crop������ͼ��
            save_path = fullfile(SAVE_Dir3,'save_tumor');   % ����SAVE��
            if ~exist(save_path,'dir')
                mkdir(save_path); % �����ļ��У�����ͼ��crop������װһ���ļ�����
            end                
            for i = 1:(d_max-d_min+1)   % crop��depth
                J=uint16(Image_crop(:,:,i));            % һ��һ��ر���ͼ�� 
                imwrite(J,fullfile(save_path,[tumor_name,num2str(j,'%04d'),'.tif']),'WriteMode','Append'); %��Ϊstack
            end
            toc
        end                                                                             
    end 
end
disp('--------------------crop�������н���--------------------');
disp('--------------------��������ʱ���£�--------------------');
toc(t0)