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

prime_number_all = primes( 10000 );

if length( prime_number_all ) < 100
  error( 'I need at least 100 prime numbers' )
end

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

% choose particular class randomly 100 of them, hopefully we have at least
% 100 samples

accuracy_all = zeros( 10, 10, 100 );

for index_trial = 1 : 100

  [ index_trial 100 ]

  for index_test_class1 = [ 1 : 10 ]
  
    for index_test_class2 = [ 1 : 10 ]
    
      if index_test_class1 ~= index_test_class2
      
        number_of_sample_per_class = 100;
        chosen_class_all = [ ( index_test_class1 - 1 ) ( index_test_class2 - 1 ) ];
        number_of_class = length( chosen_class_all );
        total_number_of_sample = number_of_sample_per_class * number_of_class;
        total_number_of_training_sample = total_number_of_sample * 0.8;
        total_number_of_test_sample = total_number_of_sample * 0.2;
        number_of_pca_component = 4;
        
        train_features_limited = zeros( 28, 28, 1, total_number_of_sample );
        train_labels_limited = zeros( total_number_of_sample, 1 );
        
        rng( prime_number_all( index_trial ) ) % For reproducibility
        
        for index_class = 1 : length( chosen_class_all )
        
          current_class = chosen_class_all( index_class );
          
          flag = YTrain == categorical( current_class );
        
          train_features_class1 = XTrain( :, :, :, flag );
        
          random_index = randperm( sum( flag ) );
        
          train_features_class2 = train_features_class1( :, :, :, random_index );
        
          train_features_limited( :, :, :, ( index_class - 1 ) * number_of_sample_per_class + [ 1 : number_of_sample_per_class ] ) = train_features_class2( :, :, :, [ 1 : number_of_sample_per_class ] );
        
          train_labels_limited( ( index_class - 1 ) * number_of_sample_per_class + [ 1 : number_of_sample_per_class ] ) = current_class;
        
        end
        
        train_labels_limited = categorical( train_labels_limited );
        
        X_train = train_features_limited;
        y_train = train_labels_limited;
        
        %% perform QPF
        Z_train = zeros( 14, 14, 4, total_number_of_sample );
        for index_image = 1 : total_number_of_sample
          
          current_XTrain = X_train( :, :, index_image );
          current_output = zeros( 14, 14, 4 );
        
          for index_j = 1 : 14
        
            for index_k = 1 : 14
        
              temp = current_XTrain( 2 * ( index_j - 1 ) + [ 1 2 ], 2 * ( index_k - 1 ) + [ 1 : 2 ] );
              current_output( index_j, index_k, : ) = circuit_two_cnots( temp( : ) );
        
            end
        
          end
        
          Z_train( :, :, :, index_image ) = current_output;
        
        end
        
        X_train = Z_train;
  
        rng( prime_number_all( index_trial ) ) % For reproducibility
        hpartition = cvpartition( y_train, 'Holdout', 0.2, 'Stratify', true );
        
        idxTrain = training(hpartition);
        sample_train = X_train( :, :, :, idxTrain );
        labels_train = y_train( idxTrain );
        idxNew = test(hpartition);
        sample_val = X_train( :, :, :, idxNew );
        labels_val = y_train( idxNew );
        
        layers = [
          imageInputLayer( [ 14 14 4 ] )
          fullyConnectedLayer( 2 )
          softmaxLayer
          classificationLayer];
  
        options = trainingOptions('adam', ...
            'ValidationData', { sample_val, labels_val }, ...
            'Verbose', false );
  
        [ net, info ] = trainNetwork( sample_train, labels_train, layers, options );
  
        accuracy_all( index_test_class1, index_test_class2, index_trial ) = info.ValidationAccuracy(end);
      
      end
  
    end
  
  end

end

mean( accuracy_all( accuracy_all ~= 0 ) )

save result_100_sample_mnist_qpf.mat accuracy_all



