clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
% myFolder = 'C:\Users\join2\Desktop\FIGSHARE\figshare data progress\figshare jpg data\3064 images';
inputFolder = 'C:\Users\join2\Desktop\FIGSHARE\figshare data progress\figshare jpg data\3064 images';
outputFolder1 = 'C:\Users\join2\Desktop\FYP\BRAIN\updated data\1';
outputFolder2 = 'C:\Users\join2\Desktop\FYP\BRAIN\updated data\2';
outputFolder3 = 'C:\Users\join2\Desktop\FYP\BRAIN\updated data\3';
fileList = dir(fullfile(inputFolder,'*.mat'));
for kk = 1:numel(fileList)
  S = load(fullfile(inputFolder,fileList(kk).name));
  I = S.cjdata.image;
  I = mat2gray(I);
  R = imresize(I,[64 64]);
  %   acessing image label from struct
  if (S.cjdata.label == 1);
        fileName = strrep(fileList(kk).name,'.mat','.png');
        imwrite(R,fullfile(outputFolder1,fileName));      
  elseif(S.cjdata.label == 2)
        fileName = strrep(fileList(kk).name,'.mat','.png');
        imwrite(R,fullfile(outputFolder2,fileName));      
  else  (S.cjdata.label == 3)
        fileName = strrep(fileList(kk).name,'.mat','.png');
        imwrite(R,fullfile(outputFolder3,fileName));

  end
 
end