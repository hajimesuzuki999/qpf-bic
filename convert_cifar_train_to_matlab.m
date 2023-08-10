clear all

label_all = zeros( 50000, 1 );
data_all = uint8(zeros( 50000, 32, 32, 3 ));

for index_file = 1 : 5

  eval( [ 'load data_batch_' num2str( index_file ) '.mat' ] )

  label_all( ( index_file - 1 ) * 10000 + [ 1 : 10000 ] ) = labels;

  data_all( ( index_file - 1 ) * 10000 + [ 1 : 10000 ], :, :, : ) = reshape( data, 10000, 32, 32, 3 );

end

XTrain = zeros( 32, 32, 1, 50000 );

for index_image = 1 : 50000

  XTrain( :, :, :, index_image ) = rgb2gray( squeeze( data_all( index_image, :, :, : ) ) );

end

YTrain = categorical( label_all );

save cifar_10_train.mat XTrain YTrain
