% batch_epc_jnd.m
clc; clear;

in_dir  = 'C:\Users\31475\Desktop\vpjnd_direction\datasets\MCL-JCI\source_images\[BMP]';          % 50张原图所在目录
out_dir = 'C:\Users\31475\Desktop\vpjnd_direction\datasets\MCL-JCI\EPC_JND_images';  % 输出JND图目录

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% 支持的图像后缀（按需增减）
exts = {'*.bmp','*.png','*.jpg','*.jpeg'};

files = [];
for k = 1:numel(exts)
    files = [files; dir(fullfile(in_dir, exts{k}))]; %#ok<AGROW>
end

fprintf('Found %d images.\n', numel(files));

for i = 1:numel(files)
    fname = files(i).name;
    fpath = fullfile(in_dir, fname);

    I = imread(fpath);

    % EPC-JND 通常用灰度计算（与原实现一致）
    if size(I,3) == 3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end

    % 计算 JND
    [~, jnd_map] = func_JND_modeling_pattern_complexity(Igray);

    % 归一化到[0,255]并转uint8，避免全黑/全白
    jnd_u8 = uint8(255 * mat2gray(jnd_map));

    % 保存成RGB 3通道（推荐，后面喂给 VP-JNDnet 更稳）
    jnd_rgb = cat(3, jnd_u8, jnd_u8, jnd_u8);

    [~, base, ~] = fileparts(fname);
    out_path = fullfile(out_dir, [base, '_EPCJND.png']);

    imwrite(jnd_rgb, out_path);

    if mod(i,10)==0 || i==numel(files)
        fprintf('Processed %d/%d: %s\n', i, numel(files), fname);
    end
end

fprintf('Done. Output in: %s\n', out_dir);