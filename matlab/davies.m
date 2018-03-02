% davies.m - pseudospectra of Davies's complex harmonic oscillator
%            with constant 1+i.  Every 10th eigenvalue is coloured red.
%
%            The L and N parameters are picked to get the best
%            results I can in standard Matlab double precision.
%
%            To run this you just need Tom Wright's Pseudospectra GUI.

  L = 14;
  N = 250;
  x = cos(pi*(0:N)/N)'; 
  c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
  X = repmat(x,1,N+1);
  dX = X-X';                  
  D  = (c*(1./c)')./(dX+(eye(N+1))); 
  D  = D - diag(sum(D'));  
  x = x(2:N);
  x = L*x; D = D/L;
  A = -D^2; A = A(2:N,2:N) + (1+1i)*diag(x.^2);
  opts.ax = [0 250 0 150];
  opts.npts=40;
  psa(A,opts);
  E = eig(A);
  [foo,ii] = sort(real(E));
  E = E(ii);
  plot(E(10:10:100),'.r','markersize',16)
