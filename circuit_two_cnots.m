function expval_all = circuit_two_cnots( phi )

index1 = [ 1 : 8 ];
index2 = [ [ 1 : 4 ] [ 9 : 12 ] ];
index3 = [ 1 2 5 6 9 10 13 14 ];
index4 = [ 1 : 2 : 15 ];

phi1_rad = pi * phi( 1 );
phi2_rad = pi * phi( 2 );
phi3_rad = pi * phi( 3 );
phi4_rad = pi * phi( 4 );

RY1 = [ [ cos( phi1_rad / 2 ), - sin( phi1_rad / 2 ) ]; [ sin( phi1_rad / 2 ), cos( phi1_rad / 2 ) ] ];
RY2 = [ [ cos( phi2_rad / 2 ), - sin( phi2_rad / 2 ) ]; [ sin( phi2_rad / 2 ), cos( phi2_rad / 2 ) ] ];
RY3 = [ [ cos( phi3_rad / 2 ), - sin( phi3_rad / 2 ) ]; [ sin( phi3_rad / 2 ), cos( phi3_rad / 2 ) ] ];
RY4 = [ [ cos( phi4_rad / 2 ), - sin( phi4_rad / 2 ) ]; [ sin( phi4_rad / 2 ), cos( phi4_rad / 2 ) ] ];

x = [ 1; zeros( 15, 1 ) ];

T1 = kron( RY1, kron( RY2, kron( RY3, RY4 ) ) );

T2 = [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]];

y = T2 * ( T1 * x );

% 
y_all = [ sum( abs( y( index1 ) ).^2 ), sum( abs( y( index2 ) ).^2 ), sum( abs( y( index3 ) ).^2 ), sum( abs( y( index4 ) ).^2 ) ];

% y_all is a probability that the outcome will be 1
expval_all = y_all + ( -1 ) * ( 1 - y_all );
