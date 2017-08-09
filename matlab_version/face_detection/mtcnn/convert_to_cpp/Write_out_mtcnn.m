% First writing out PNet
load('../PNet_mlab.mat');

cnn = struct;
cnn.layers = cell(1,8);
cnn.layers{1} = struct;
cnn.layers{1}.type = 'conv';
cnn.layers{1}.weights = {PNet_mlab.weights_conv1, PNet_mlab.biases_conv1};

cnn.layers{2} = struct;
cnn.layers{2}.type = 'prelu';
cnn.layers{2}.weights = {PNet_mlab.prelu_weights_1};

cnn.layers{3} = struct;
cnn.layers{3}.type = 'max_pooling';
cnn.layers{3}.weights = {};
cnn.layers{3}.stride_x = 2;
cnn.layers{3}.stride_y = 2;
cnn.layers{3}.kernel_size_x = 2;
cnn.layers{3}.kernel_size_y = 2;

cnn.layers{4} = struct;
cnn.layers{4}.type = 'conv';
cnn.layers{4}.weights = {PNet_mlab.weights_conv2, PNet_mlab.biases_conv2};

cnn.layers{5} = struct;
cnn.layers{5}.type = 'prelu';
cnn.layers{5}.weights = {PNet_mlab.prelu_weights_2};

cnn.layers{6} = struct;
cnn.layers{6}.type = 'conv';
cnn.layers{6}.weights = {PNet_mlab.weights_conv3, PNet_mlab.biases_conv3};

cnn.layers{7} = struct;
cnn.layers{7}.type = 'prelu';
cnn.layers{7}.weights = {PNet_mlab.prelu_weights_3};

cnn.layers{8} = struct;
cnn.layers{8}.type = 'fc';
cnn.layers{8}.weights = {PNet_mlab.w, PNet_mlab.b};

Write_CNN_to_binary('PNet.dat', cnn);