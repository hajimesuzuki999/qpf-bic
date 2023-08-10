% Please download the EMNIST data (Matlab version, emnist-matlab.zip) from
% https://www.westernsydney.edu.au/icns/resources/reproducible_research3/publication_support_materials2/emnist
% and extract emnist-balanced.mat.  Keep it in the current folder.

clear all

prime_number_all = primes( 10000 );

if length( prime_number_all ) < 100
  error( 'I need at least 100 prime numbers' )
end

load emnist-balanced.mat

number_of_train_images = 112800;
number_of_test_images = 18800;

% 47 classes, 0 to 46
YTrain = categorical( dataset.train.labels );
XTrain = reshape( ( double( dataset.train.images ) ).', 28, 28, 1, number_of_train_images ) / 255;

YTest = categorical( dataset.test.labels );
XTest = reshape( ( double( dataset.test.images ) ).', 28, 28, 1, number_of_test_images ) / 255;

% perform QPF
ZTrain = zeros( 14, 14, 4, number_of_train_images );
for index_image = 1 : number_of_train_images
  
  current_XTrain = XTrain( :, :, index_image );
  current_output = zeros( 14, 14, 4 );

  for index_j = 1 : 14

    for index_k = 1 : 14

      temp = current_XTrain( 2 * ( index_j - 1 ) + [ 1 2 ], 2 * ( index_k - 1 ) + [ 1 : 2 ] );
      current_output( index_j, index_k, : ) = circuit_two_cnots( temp( : ) );

    end

  end

  ZTrain( :, :, :, index_image ) = current_output;

end
XTrain = ZTrain;

ZTest = zeros( 14, 14, 4, number_of_test_images );
for index_image = 1 : number_of_test_images
  
  current_XTest = XTest( :, :, index_image );
  current_output = zeros( 14, 14, 4 );

  for index_j = 1 : 14

    for index_k = 1 : 14

      temp = current_XTest( 2 * ( index_j - 1 ) + [ 1 2 ], 2 * ( index_k - 1 ) + [ 1 : 2 ] );
      current_output( index_j, index_k, : ) = circuit_two_cnots( temp( : ) );

    end

  end

  ZTest( :, :, :, index_image ) = current_output;

end
XTest = ZTest;

% choose particular class randomly 100 of them, hopefully we have at least
% 100 samples

accuracy_all = zeros( 47, 47, 100 );

tic

for index_trial = 1 : 100

  [ index_trial 100 ]

  for index_test_class1 = [ 1 : 47 ]
  
    for index_test_class2 = [ 1 : 47 ]
    
      if index_test_class1 ~= index_test_class2
      
        number_of_sample_per_class = 100;
        chosen_class_all = [ ( index_test_class1 - 1 ) ( index_test_class2 - 1 ) ];
        number_of_class = length( chosen_class_all );
        total_number_of_sample = number_of_sample_per_class * number_of_class;
        total_number_of_training_sample = total_number_of_sample * 0.8;
        total_number_of_test_sample = total_number_of_sample * 0.2;
        
        train_features_limited = zeros( 14, 14, 4, total_number_of_sample );
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

toc

mean( accuracy_all( accuracy_all ~= 0 ) )

save result_100_sample_emnist_qpf.mat accuracy_all


