clear all

load result_100_sample_mnist_nn.mat accuracy_all

mean_accuracy_all = zeros( 100, 1 );
variance_accuracy_all = zeros( 100, 1 );

% create index excluding that corresponds to ( 1, 1 ), ( 2, 2 ), etc.
index_all = [ 1 : 100 ].';
index_remove = zeros( 10, 1 );

for index_diagonal = 1 : 10

  index_remove( index_diagonal ) = sub2ind( [ 10 10 ], index_diagonal, index_diagonal );

end

index_removed = setxor( index_all, index_remove );

for index_trial = 1 : 100

 a = accuracy_all( :, :, index_trial );
 b = a( index_removed );

 mean_accuracy_all( index_trial ) = mean( b );
 variance_accuracy_all( index_trial ) = var( b );

end

mean_accuracy_all_classical = mean_accuracy_all;

%% quantum

load result_100_sample_mnist_qpf.mat accuracy_all

mean_accuracy_all = zeros( 100, 1 );
variance_accuracy_all = zeros( 100, 1 );

% create index excluding that corresponds to ( 1, 1 ), ( 2, 2 ), etc.
index_all = [ 1 : 100 ].';
index_remove = zeros( 10, 1 );

for index_diagonal = 1 : 10

  index_remove( index_diagonal ) = sub2ind( [ 10 10 ], index_diagonal, index_diagonal );

end

index_removed = setxor( index_all, index_remove );

for index_trial = 1 : 100

 a = accuracy_all( :, :, index_trial );
 b = a( index_removed );

 mean_accuracy_all( index_trial ) = mean( b );
 variance_accuracy_all( index_trial ) = var( b );

end

mean_accuracy_all_quantum = mean_accuracy_all;

mean( mean_accuracy_all_classical )
mean( mean_accuracy_all_quantum )

%%

figure( 1 )

clf

set(gcf,'DefaultAxesFontName','Times');
set(gcf,'DefaultAxesFontSize',10);
set(gcf,'DefaultTextFontName','Times');
set(gcf,'DefaultTextFontSize',10);
set(gcf,'PaperOrientation','portrait')
set(gcf,'PaperType','A4')
set(gcf,'PaperUnits','Inches')
set(gcf,'PaperPosition',[0.25 0.25 8.5/2.54*2.1 8.5/2.54*1.3])
set(gcf,'Position',[50 50 round(8.5/2.45*96*2.1) round(8.5/2.54*96*1.3)])
set(gcf,'Color',[1 1 1])

plot( ...
  [ 1 : 100 ], mean_accuracy_all_classical, 'b.-', ...
  [ 1 : 100 ], mean_accuracy_all_quantum, 'r.-' )

grid on
box on

xlabel( 'Trial index' )
ylabel( 'Testing accuracy (%)' )

legend(...
  'NN', ...
  'QPF-NN', ...
  'location', 'southeast' )

legend boxoff

print -dpng -r300 plot_accuracy_100_sample_mnist.png


