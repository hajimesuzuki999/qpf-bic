% Please download the following MNIST data from
% http://yann.lecun.com/exdb/mnist/, unzip them, and keep them in the
% current directory.
%
% train-images-idx3-ubyte.gz
% train-labels-idx1-ubyte.gz
% t10k-images-idx3-ubyte.gz
% t10k-labels-idx1-ubyte.gz
%

clear all

%% load MNIST dataset

fid = fopen( 'train-labels.idx1-ubyte', 'r' );

magic_number = fread( fid, 1, 'int32', 'b' )
number_of_items = fread( fid, 1, 'int32', 'b' )

dat = fread( fid, number_of_items, 'uint8', 'b' );

fclose( fid );

YTrain = categorical( dat );

fid = fopen( 'train-images.idx3-ubyte', 'r' );

magic_number = fread( fid, 1, 'int32', 'b' )
number_of_train_images = fread( fid, 1, 'int32', 'b' )
number_of_rows = fread( fid, 1, 'int32', 'b' )
number_of_columns = fread( fid, 1, 'int32', 'b' )

dat = fread( fid, number_of_rows * number_of_columns * number_of_train_images, 'uint8', 'b' );

fclose( fid );

XTrain = reshape( dat, number_of_rows, number_of_columns, 1, number_of_train_images ) / 255;

% check number of images in each class
number_of_train_images_per_class = zeros( 10, 1 );
for index_digit = 0 : 9
  number_of_train_images_per_class( index_digit + 1 ) = sum( YTrain == categorical( index_digit ) );
end

fid = fopen( 't10k-labels.idx1-ubyte', 'r' );

magic_number = fread( fid, 1, 'int32', 'b' )
number_of_items = fread( fid, 1, 'int32', 'b' )

dat = fread( fid, number_of_items, 'uint8', 'b' );

fclose( fid );

YTest = categorical( dat );

fid = fopen( 't10k-images.idx3-ubyte', 'r' );

disp( 'magic number should be 2051' )

magic_number = fread( fid, 1, 'int32', 'b' )
number_of_test_images = fread( fid, 1, 'int32', 'b' )
number_of_rows = fread( fid, 1, 'int32', 'b' )
number_of_columns = fread( fid, 1, 'int32', 'b' )

dat = fread( fid, number_of_rows * number_of_columns * number_of_test_images, 'uint8', 'b' );

fclose( fid );

XTest = reshape( dat, number_of_rows, number_of_columns, 1, number_of_test_images ) / 255;

accuracy_all = zeros( 10, 10 );

for index_test_class1 = [ 1 : 10 ]

  for index_test_class2 = [ 1 : 10 ]

    [ index_test_class1 index_test_class2 ]
    
    if index_test_class1 ~= index_test_class2
    
      chosen_class_all = [ ( index_test_class1 - 1 ) ( index_test_class2 - 1 ) ];
      number_of_class = length( chosen_class_all );
                        
      flag1 = YTrain == categorical( chosen_class_all( 1 ) );
      train_features_class1 = XTrain( :, :, :, flag1 );      
      number_of_training_sample_class1 = size( train_features_class1, 4 );
      flag2 = YTrain == categorical( chosen_class_all( 2 ) );
      train_features_class2 = XTrain( :, :, :, flag2 );
      number_of_training_sample_class2 = size( train_features_class2, 4 );
      total_number_of_training_sample = number_of_training_sample_class1 + number_of_training_sample_class2;
      train_features_limited = zeros( 28, 28, 1, total_number_of_training_sample );      
      train_labels_limited = zeros( total_number_of_training_sample, 1 );
      train_features_limited( :, :, :, [ 1 : number_of_training_sample_class1 ] ) = train_features_class1;
      train_features_limited( :, :, :, number_of_training_sample_class1 + [ 1 : number_of_training_sample_class2 ] ) = train_features_class2;
      train_labels_limited( 1 : number_of_training_sample_class1 ) = chosen_class_all( 1 );
      train_labels_limited( number_of_training_sample_class1 + [ 1 : number_of_training_sample_class2 ] ) = chosen_class_all( 2 );
      train_labels_limited = categorical( train_labels_limited );
      
      flag1 = YTest == categorical( chosen_class_all( 1 ) );
      test_features_class1 = XTest( :, :, :, flag1 );      
      number_of_testing_sample_class1 = size( test_features_class1, 4 );
      flag2 = YTest == categorical( chosen_class_all( 2 ) );
      test_features_class2 = XTest( :, :, :, flag2 );
      number_of_testing_sample_class2 = size( test_features_class2, 4 );
      total_number_of_testing_sample = number_of_testing_sample_class1 + number_of_testing_sample_class2;
      test_features_limited = zeros( 28, 28, 1, total_number_of_testing_sample );      
      test_labels_limited = zeros( total_number_of_testing_sample, 1 );
      test_features_limited( :, :, :, [ 1 : number_of_testing_sample_class1 ] ) = test_features_class1;
      test_features_limited( :, :, :, number_of_testing_sample_class1 + [ 1 : number_of_testing_sample_class2 ] ) = test_features_class2;
      test_labels_limited( 1 : number_of_testing_sample_class1 ) = chosen_class_all( 1 );
      test_labels_limited( number_of_testing_sample_class1 + [ 1 : number_of_testing_sample_class2 ] ) = chosen_class_all( 2 );      
      test_labels_limited = categorical( test_labels_limited );

      sample_train = train_features_limited;
      labels_train = train_labels_limited;

      sample_val = test_features_limited;
      labels_val = test_labels_limited;
      
      layers = [
        imageInputLayer( [ 28 28 1 ] )
        fullyConnectedLayer( 2 )
        softmaxLayer
        classificationLayer];

      options = trainingOptions('adam', ...
          'ValidationData', { sample_val, labels_val }, ...
          'Verbose', false );

      [ net, info ] = trainNetwork( sample_train, labels_train, layers, options );

      accuracy_all( index_test_class1, index_test_class2 ) = info.ValidationAccuracy(end);
    
    end

  end

end

mean( accuracy_all( accuracy_all ~= 0 ) )

save classify_all_data_mnist_nn.mat accuracy_all



