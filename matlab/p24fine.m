% p24fine.m - pseudospectra of Davies's complex harmonic oscillator
%             (This is the finer, slower plot, with 0:2 changed to 0:.5.)

% Eigenvalues:
  N = 70; [D,x] = cheb(N); x = x(2:N);
  L = 6; x = L*x; D = D/L;                   % rescale to [-L,L]
  A = -D^2; A = A(2:N,2:N) + (1+3i)*diag(x.^2);
  clf, plot(eig(A),'.','markersize',14)
  axis([0 50 0 40]), drawnow, hold on
  
% Pseudospectra:
  x = 0:.5:50; y = 0:.5:40; [xx,yy] = meshgrid(x,y); zz = xx+1i*yy;
  I = eye(N-1); sigmin = zeros(length(y),length(x));
  h = waitbar(0,'please wait...');
  tic
  for j = 1:length(x), waitbar(j/length(x))
    for i = 1:length(y), sigmin(i,j) = min(svd(zz(i,j)*I-A)); end
  end, close(h)
  toc
  contour(x,y,sigmin,10.^(-4:.5:-.5)), colormap(1e-6*[1 1 1]);
