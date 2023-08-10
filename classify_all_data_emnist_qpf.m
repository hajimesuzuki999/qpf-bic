% Please download the EMNIST data (Matlab version, emnist-matlab.zip) from
% https://www.westernsydney.edu.au/icns/resources/reproducible_research3/publication_support_materials2/emnist
% and extract emnist-balanced.mat.  Keep it in the current folder.

clear all

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

accuracy_all = zeros( 47, 47 );

%%

for index_test_class1 = [ 1 : 47 ]

  for index_test_class2 = [ 1 : 47 ]

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
      train_features_limited = zeros( 14, 14, 4, total_number_of_training_sample );      
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
      test_features_limited = zeros( 14, 14, 4, total_number_of_testing_sample );      
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
        imageInputLayer( [ 14 14 4 ] )
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

save result_all_data_emnist_qpf.mat accuracy_all


