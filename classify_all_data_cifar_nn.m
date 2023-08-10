% Please download the CIFAR-10 data (Matlab version) from
% https://www.cs.toronto.edu/~kriz/cifar.html, unzip them, and use convert_cifar_test_to_matlab.m 
% and convert_cifar_train_to_matlab.m to create cifar_10_test.mat and cifar_10_train.mat used in this script.  
% Keep them in the current directory.
%

clear all

load cifar_10_train.mat XTrain YTrain

XTrain = XTrain / 255;

% check number of images in each class
number_of_train_images_per_class = zeros( 10, 1 );
for index_digit = 0 : 9
  number_of_train_images_per_class( index_digit + 1 ) = sum( YTrain == categorical( index_digit ) );
end

load cifar_10_test.mat XTest YTest

XTest = XTest / 255;

accuracy_all = zeros( 10, 10 );

tic

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
      train_features_limited = zeros( 32, 32, 1, total_number_of_training_sample );      
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
      test_features_limited = zeros( 32, 32, 1, total_number_of_testing_sample );      
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
        imageInputLayer( [ 32 32 1 ] )
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

toc

mean( accuracy_all( accuracy_all ~= 0 ) )

save result_all_data_cifar_nn.mat accuracy_all


