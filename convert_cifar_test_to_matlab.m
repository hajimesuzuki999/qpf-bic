clear all

load test_batch.mat

data_all = uint8( reshape( data, 10000, 32, 32, 3 ) );

XTest = zeros( 32, 32, 1, 10000 );

for index_image = 1 : 10000

  XTest( :, :, :, index_image ) = rgb2gray( squeeze( data_all( index_image, :, :, : ) ) );

end

YTest = categorical( labels );

save cifar_10_test.mat XTest YTest
