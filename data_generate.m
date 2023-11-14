rng(2023);
num_BS = 7;
num_User = 30;
num_H = 1000;
R = 100;
minR_ratio = 0.2;
try_seed = 1;
var_noise = 1;
% [7,6] - [2.2977, 7.7636]; [7,4] - [2.2974, 7.7322]; [5,6] - [2.8789, 9.1013]; [5,4] -
% [2.8882, 9.0865];
[H, ~, Y] = Generate_IMAC_function(num_BS, num_User, num_H, R, minR_ratio, try_seed, var_noise);
X = H(1:num_User:end,:,:);
% whos X
file_name = ['IMAC_BS',num2str(num_BS),'_User',num2str(num_User),'_Channel',num2str(num_H),'_R',num2str(R),'Ratio',num2str(minR_ratio),'.mat'];
save(file_name, 'X','Y'); 
% save(file_name, 'Y');

