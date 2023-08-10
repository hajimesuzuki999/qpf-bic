% Please download the CIFAR-10 data (Matlab version) from
% https://www.cs.toronto.edu/~kriz/cifar.html, unzip them, and use convert_cifar_test_to_matlab.m 
% and convert_cifar_train_to_matlab.m to create cifar_10_test.mat and cifar_10_train.mat used in this script.  
% Keep them in the current directory.
%

clear all

prime_number_all = primes( 10000 );

if length( prime_number_all ) < 100
  error( 'I need at least 100 prime numbers' )
end

load cifar_10_train.mat XTrain YTrain

XTrain = XTrain / 255;

load cifar_10_test.mat XTest YTest

XTest = XTest / 255;

% choose particular class randomly 100 of them, hopefully we have at least
% 100 samples

accuracy_all = zeros( 10, 10, 100 );

for index_trial = 1 : 100

  for index_test_class1 = [ 1 : 10 ]
  
    for index_test_class2 = [ 1 : 10 ]
  
      [ index_test_class1 index_test_class2 ]
  
      if index_test_class1 ~= index_test_class2
      
        number_of_sample_per_class = 100;
        chosen_class_all = [ ( index_test_class1 - 1 ) ( index_test_class2 - 1 ) ];
        number_of_class = length( chosen_class_all );
        total_number_of_sample = number_of_sample_per_class * number_of_class;
        total_number_of_training_sample = total_number_of_sample * 0.8;
        total_number_of_test_sample = total_number_of_sample * 0.2;
        
        train_features_limited = zeros( 32, 32, 1, total_number_of_sample );
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
          imageInputLayer( [ 32 32 1 ] )
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

save result_100_sample_cifar_nn.mat accuracy_all


