% 设置路径
folderPath = 'F:\matlab\机器学习课程\作业\期末作业\code\fold'; 
classFile = 'class.xlsx'; 

% 读取 class.xlsx 文件
classData = readtable(classFile);

% 获取所有图片文件名和类别标签
fileNames = classData{:, 1};  % 图片文件名
labels = classData{:, 2};  % 对应类别标签（0、2、3、4、9）

% 预定义类别和划分比例
uniqueLabels = unique(labels);
trainImages = [];  % 训练集图片
testImages = [];  % 测试集图片
trainLabels = [];  % 训练集标签
testLabels = [];  % 测试集标签
%%
% 设置固定的随机种子
rng(0); 

% 按类别划分训练集和测试集
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);
    
    % 找到该标签下的所有图片
    indices = find(labels == label);
    
    % 随机打乱图片顺序
    selectedFiles = fileNames(indices);
    selectedFiles = selectedFiles(randperm(length(selectedFiles)));
    
    % 从每个类别中选取60张训练集，20张测试集
    trainFiles = selectedFiles(1:800);
    testFiles = selectedFiles(801:1000);
    
    % 将选中的图片添加到训练集和测试集
    for j = 1:length(trainFiles)
        img = imread(fullfile(folderPath, trainFiles{j}));
        img = imresize(img, [224, 224]);  % 调整图像大小为224x224
        img = single(img) / 255;  % 将图像像素值归一化到[0, 1]
        trainImages = cat(4, trainImages, img);  % 存储图像，使用4D数组
        trainLabels = [trainLabels; label];
    end
    
    for j = 1:length(testFiles)
        img = imread(fullfile(folderPath, testFiles{j}));
        img = imresize(img, [224, 224]);  % 调整图像大小为224x224
        img = single(img) / 255;  % 将图像像素值归一化到[0, 1]
        testImages = cat(4, testImages, img);  % 存储图像，使用4D数组
        testLabels = [testLabels; label];
    end
end

% 转换标签为 categorical 格式
trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);
%%
% 加载预训练的 GoogLeNet 模型
net = googlenet;

% 获取网络的层
layers = net.Layers;

% 创建 layerGraph 对象
lgraph = layerGraph(net);

% 替换全连接层
% GoogLeNet 的最后一层全连接层是 'loss3-classifier'
newFCLayer = fullyConnectedLayer(5, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraph = replaceLayer(lgraph, 'loss3-classifier', newFCLayer);

% 替换分类层
% GoogLeNet 的分类层是 'classoutput'
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, 'output', newClassLayer);
%%
% 确保输入图像的大小为 224x224x3
inputSize = [224 224 3];
% 确保数据类型正确
trainImages = single(trainImages);  % 转换为 single 类型
testImages = single(testImages);  % 转换为 single 类型
% 记录训练开始时间
trainStartTime = tic;
% 保存训练损失和训练准确度
trainLoss = [];
trainAccuracy = [];
% 设置训练选项
options = trainingOptions('sgdm', ...
    'MaxEpochs', 4, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.0001, ...  % 设置较小的学习率
    'Plots', 'training-progress', ...  % 训练进度对话框
    'Verbose', false, ...
    'OutputFcn', @(info) saveProgress(info, trainLoss, trainAccuracy));

% 训练模型
netTransfer = trainNetwork(trainImages, trainLabels, lgraph, options);

% 记录训练结束时间
trainEndTime = toc(trainStartTime);
fprintf('训练阶段运行时间：%.2f秒\n', trainEndTime);
%%
% 记录分类开始时间
classifyStartTime = tic;

% 使用训练好的模型进行预测
predictedLabels = classify(netTransfer, testImages);

% 记录分类结束时间
classifyEndTime = toc(classifyStartTime);
fprintf('分类阶段运行时间：%.2f秒\n', classifyEndTime);
%%
% 计算分类精度
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
fprintf('分类精度：%.2f%%\n', accuracy * 100);

% 计算混淆矩阵
confMat = confusionmat(testLabels, predictedLabels);
disp('混淆矩阵:');
disp(confMat);

% 计算每个类别的用户精度
userAccuracy = diag(confMat) ./ sum(confMat, 2);  % 用户精度 = 每类正确分类数 / 每类真实样本数
fprintf('每类的用户精度（准确度）:\n');
disp(userAccuracy);

% 计算观察到的一致性比率（po）
po = sum(diag(confMat)) / sum(confMat(:));

% 计算预期的一致性比率（pe）
rowSums = sum(confMat, 2);
colSums = sum(confMat, 1);
pe = sum(rowSums .* colSums) / (sum(confMat(:))^2);

% 计算Kappa系数
kappa = (po - pe) / (1 - pe);
fprintf('Kappa系数：%.4f\n', kappa);
%%
% 保存训练进度中的曲线图
function saveProgress(info, trainLoss, trainAccuracy)
    persistent lossData accuracyData;    
    if isfield(info, 'TrainingLoss') && ~isempty(info.TrainingLoss)
        lossData = [lossData; info.TrainingLoss];  % 保存损失数据
    end    
    if isfield(info, 'TrainingAccuracy') && ~isempty(info.TrainingAccuracy)
        accuracyData = [accuracyData; info.TrainingAccuracy];  % 保存准确度数据
    end
    
    % 在训练结束时，保存曲线图
    if info.State == "done"
        figure('Position', [100, 100, 800, 600], 'PaperPositionMode', 'auto');  % 设置图形窗口的位置和大小
        
        % 绘制训练准确度曲线
        subplot(2, 1, 1);  % 将子图排列为2行1列的第一个
        plot(accuracyData, 'LineWidth', 1, 'Color', [0/255, 114/255, 189/255]);  % 设置准确度曲线的颜色
        title('Training Accuracy');
        xlabel('Iteration');
        ylabel('Accuracy (%)');
        xlim([0, 500]);  % 设置横坐标范围为0到500
        
        % 绘制训练损失曲线
        subplot(2, 1, 2);  % 将子图排列为2行1列的第二个
        plot(lossData, 'LineWidth', 1, 'Color', [217/255, 83/255, 25/255]);  % 设置损失曲线的颜色
        title('Training Loss');
        xlabel('Iteration');
        ylabel('Loss');
        xlim([0, 500]);  % 设置横坐标范围为0到500
        
        % 保存为图片，使用较高的分辨率
        set(gcf, 'PaperUnits', 'inches', 'PaperSize', [8, 6]);
        print(gcf, 'googlenet.png', '-dpng', '-r300');  % 使用300 DPI的分辨率保存
    end
end