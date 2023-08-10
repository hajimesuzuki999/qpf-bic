clear all

load result_all_data_mnist_qpf.mat accuracy_all

figure( 1 )

clf

set(gcf,'DefaultAxesFontName','Times');
set(gcf,'DefaultAxesFontSize',10);
set(gcf,'DefaultTextFontName','Times');
set(gcf,'DefaultTextFontSize',10);
set(gcf,'PaperOrientation','portrait')
set(gcf,'PaperType','A4')
set(gcf,'PaperUnits','Inches')
set(gcf,'PaperPosition',[0.25 0.25 8.5/2.54*1.4 8.5/2.54*1.3])
set(gcf,'Position',[50 50 round(8.5/2.45*96*1.4) round(8.5/2.54*96*1.3)])
set(gcf,'Color',[1 1 1])

imagesc( accuracy_all )
set( gca, 'xaxisLocation', 'top' )
set( gca, 'xtick', [ 1 : 10 ] )
set( gca, 'ytick', [ 1 : 10 ] )
set( gca, 'xticklabel', [ 0 : 9 ] )
set( gca, 'yticklabel', [ 0 : 9 ] )

h = colorbar;

caxis( [ 95 100 ] )

set( h, 'Ticks', [ 95 : 100 ] )

print -dpng -r300 map_accuracy_all_data_mnist_qpf.png


